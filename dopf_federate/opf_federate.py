import logging
import helics as h
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Union
from oedisi.types.common import BrokerConfig
from oedisi.types.data_types import (
    CommandList,
    Command,
    PowersImaginary,
    PowersReal,
    Injection,
    Topology,
    VoltagesMagnitude,
)
from pydantic import BaseModel
import dopfpso

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class AlgorithmParameters(BaseModel):
    tol: float = 5e-7
    base_power: Optional[float] = 100.0

    class Config:
        use_enum_values = True

class OPFFederate:
    "OPF federate. Wraps OPF with pubs and subs"

    def __init__(
        self,
        federate_name,
        algorithm_parameters: AlgorithmParameters,
        input_mapping,
        broker_config: BrokerConfig,
    ):
        "Initializes federate with name and remaps input into subscriptions"
        deltat = 1

        self.algorithm_parameters = algorithm_parameters

        # Create Federate Info object that describes the federate properties #
        fedinfo = h.helicsCreateFederateInfo()

        h.helicsFederateInfoSetBroker(fedinfo, broker_config.broker_ip)
        h.helicsFederateInfoSetBrokerPort(fedinfo, broker_config.broker_port)

        fedinfo.core_name = federate_name
        fedinfo.core_type = h.HELICS_CORE_TYPE_ZMQ
        fedinfo.core_init = "--federates=1"
        h.helicsFederateInfoSetTimeProperty(
            fedinfo, h.helics_property_time_delta, deltat
        )

        self.vfed = h.helicsCreateValueFederate(federate_name, fedinfo)
        logger.info("Value federate created")

    
        # Register the publication #
        self.sub_power_P = self.vfed.register_subscription(
            input_mapping["powers_real"], "W"
        )
        self.sub_power_Q = self.vfed.register_subscription(
            input_mapping["powers_imag"], "W"
        )
        self.sub_topology = self.vfed.register_subscription(
            input_mapping["topology"], ""
        )
        self.pub_voltages = self.vfed.register_publication(
            "opf_voltages_magnitude", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_commands = self.vfed.register_publication(
            "pv_set", h.HELICS_DATA_TYPE_STRING, ""
        )

    def run(self):

        logger.info(f"Federate connected: {datetime.now()}")
        self.vfed.enter_executing_mode()
        granted_time = h.helicsFederateRequestTime(self.vfed, h.HELICS_TIME_MAXTIME)
        t = 0
        while granted_time < h.HELICS_TIME_MAXTIME:
            if not self.sub_power_P.is_updated():
                granted_time = h.helicsFederateRequestTime(
                    self.vfed, h.HELICS_TIME_MAXTIME
                )
                continue

            load_p = PowersReal.parse_obj(self.sub_power_P.json)
            load_q = PowersImaginary.parse_obj(self.sub_power_Q.json)
            logger.info(f'load_p:{load_p.values}')
            logger.info(f'load_q:{load_q.values}')
            time = load_p.time
            logger.info(f'time:{time}')

            pv_reactive, bus_voltages = dopfpso.dopf_step(load_p,load_q,t)
            t = t+1

            commands = []
            for n in range(len(pv_reactive['values'])):
                commands.append((pv_reactive['eqid'][n], 0, pv_reactive['values'][n]))

            if commands:
                self.pub_commands.publish(
                    json.dumps(commands)
                )
            logger.info(f"commands: {commands}") 
            # logger.info(f"ids: {bus_voltages['ids']}") 
            logger.info(f"values: {bus_voltages['values']}")            
            pub_mags = VoltagesMagnitude(ids=bus_voltages['ids'], values=bus_voltages['values'], time=time)
            self.pub_voltages.publish(
                pub_mags.json()
            )
            logger.info(f"pub_mags: {pub_mags}") 

        self.stop()

    def stop(self):
        h.helicsFederateDisconnect(self.vfed)
        h.helicsFederateFree(self.vfed)
        h.helicsCloseLibrary()
        logger.info(f"Federate disconnected: {datetime.now()}")


def run_simulator(broker_config: BrokerConfig):
    logger.info(f"Running---------------------------------------------------") 
    with open("static_inputs.json") as f:
        config = json.load(f)
        federate_name = config["name"]
        if "algorithm_parameters" in config:
            parameters = AlgorithmParameters.parse_obj(config["algorithm_parameters"])
        else:
            parameters = AlgorithmParameters.parse_obj({})

    with open("input_mapping.json") as f:
        input_mapping = json.load(f)

    sfed = OPFFederate(
        federate_name, parameters, input_mapping, broker_config
    )

    try:
        sfed = OPFFederate(
        federate_name, parameters, input_mapping, broker_config
        )
        logger.info("Value federate created")
    except h.HelicsException as e:
        logger.error(f"Failed to create HELICS Value Federate: {str(e)}")
        
    sfed.run()
    logger.info(f"Running------------------------------------------------") 

if __name__ == "__main__":
    run_simulator(BrokerConfig(broker_ip="127.0.0.1"))