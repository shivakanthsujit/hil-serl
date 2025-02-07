from experiments.ram_insertion.config import TrainConfig as RAMInsertionTrainConfig
from experiments.usb_pickup_insertion.config import TrainConfig as USBPickupInsertionTrainConfig
from experiments.object_handover.config import TrainConfig as ObjectHandoverTrainConfig
from experiments.egg_flip.config import TrainConfig as EggFlipTrainConfig
from experiments.pick_cube_sim.config import TrainConfig as PickCubeSimTrainConfig
from experiments.pick_cube_sim.config import TrainConfigBin as BinSimTrainConfig
from experiments.robohive_pick.config import TrainConfig as RobohivePickTrainConfig

CONFIG_MAPPING = {
                "ram_insertion": RAMInsertionTrainConfig,
                "usb_pickup_insertion": USBPickupInsertionTrainConfig,
                "object_handover": ObjectHandoverTrainConfig,
                "egg_flip": EggFlipTrainConfig,
                "pick_cube_sim": PickCubeSimTrainConfig,
                "bin_sim": BinSimTrainConfig,
                "robohive_pick": RobohivePickTrainConfig,
               }
