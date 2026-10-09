import os
import roboticstoolbox as rtb
from ir_support.robots import UR3e
from spatialgeometry import Mesh
import swift
import numpy as np
import matplotlib.pyplot as plt
from roboticstoolbox import DHLink, DHRobot
from math import pi
from ir_support import CylindricalDHRobotPlot
from ir_support_extra_robots.robots import Turtlebot3Waffle
from ir_support_extra_parts.parts import part_mesh
from spatialmath import SE3
from spatialmath.base import transl, trotx, troty, trotz, tr2rpy
from roboticstoolbox import jtraj
from ir_support import tranimate_custom
from UR3 import add_UR3, UR3_move_base, UR3_move_and_grab, set_UR3_pose

env = swift.Swift()
env.launch(realtime=True)
item_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Scene_parts")

#cognac scene load
cognac_location = os.path.join(item_folder, "Cognac.stl")
cognac_scene = Mesh(filename=cognac_location, color="#522B0F", scale=[0.001, 0.001, 0.001])
cognac_scene.T = SE3(0.3, -0.25, 0.375) 
env.add(cognac_scene)

#cointreau.stl
cointreau_location = os.path.join(item_folder, "Cointreau.stl")
cointreau_scene = Mesh(filename=cointreau_location, color="#C47C07", scale=[0.02, 0.02, 0.02])
cointreau_scene.T = SE3(0.3, -0.2, 0.25)
env.add(cointreau_scene)

#lemonSlice.stl 
lemon_location = os.path.join(item_folder, "Lemon.stl")
lemon_scene = Mesh(filename=lemon_location, color="#F7F7A1")
lemon_scene.T = SE3(0.22, -0.2, 0.26) * troty(pi/2)
env.add(lemon_scene)

#BarTableRobot.stl
bar_table_location = os.path.join(item_folder, "BarTableRobot.stl")
bar_table_scene = Mesh(filename=bar_table_location, color="#8B4513", scale=[0.001, 0.001, 0.001])
bar_table_scene.T = SE3(-1.5, 0.5, 0)
env.add(bar_table_scene)

#SensorsTable
sensors_table_location = os.path.join(item_folder, "SensorsTable.stl")
sensors_table_scene = Mesh(filename=sensors_table_location, color="#8B4513", scale=[0.001, 0.001, 0.001])
sensors_table_scene.T = SE3(1.85,0,0) * trotz(pi/2) * SE3(0,0.25,0)
env.add(sensors_table_scene)

#BarTableStorage.stl
bar_table_storage_location = os.path.join(item_folder, "BarTableStorage.stl")
bar_table_storage_scene = Mesh(filename=bar_table_storage_location, color="#8B4513", scale=[0.001, 0.001, 0.001])
bar_table_storage_scene.T = SE3(-0.5, -0.5, 0) 
env.add(bar_table_storage_scene)

#BetterLemon.stl
Better_lemon_location = os.path.join(item_folder, "BetterLemon.stl")
Better_lemon_scene = Mesh(filename=Better_lemon_location, color="#F7F7A1", scale=[0.001, 0.001, 0.001])
Better_lemon_scene.T = SE3(0.2, -0.2, 0.26) * troty(pi/2)
env.add(Better_lemon_scene)

#UR3 rail
ur3_rail_location = os.path.join(item_folder, "RailForUR3.stl")
ur3_rail_scene = Mesh(filename=ur3_rail_location, color="#C0C0C0", scale=[0.001, 0.001, 0.001])
ur3_rail_scene.T = SE3(-1.5, 0.1, 0)
env.add(ur3_rail_scene)

#cocktail glass
cocktail_glass_location = os.path.join(item_folder, "CocktailGlass.stl")
cocktail_glass_scene = Mesh(filename=cocktail_glass_location, color="#FFFFFF", scale=[0.001, 0.001, 0.001])
cocktail_glass_scene.T = SE3(-0.2, -0.1, 0.255)
env.add(cocktail_glass_scene)

#cocktail shaker
cocktail_shaker_location = os.path.join(item_folder, "CocktailShaker.stl")
cocktail_shaker_scene = Mesh(filename=cocktail_shaker_location, color="#504E4E", scale=[0.001, 0.001, 0.001])
cocktail_shaker_scene.T = SE3(0, -0.2, 0.255)
env.add(cocktail_shaker_scene)

#Umbrella pot
Umbrella_pot_location = os.path.join(item_folder, "UmbrellaPot.stl")
Umbrella_pot_scene = Mesh(filename=Umbrella_pot_location, color="#504E4E", scale=[0.001, 0.001, 0.001])
Umbrella_pot_scene.T = SE3(-0.2, -0.3, 0.255)
env.add(Umbrella_pot_scene)

#Umbrella 
Umbrella_location = os.path.join(item_folder, "Umbrella.stl")
Umbrella_scene = Mesh(filename=Umbrella_location, color="#F601C1", scale=[0.001, 0.001, 0.001])
Umbrella_scene.T = SE3(-0.2, -0.3, 0.255) * trotz(pi/2)
env.add(Umbrella_scene)

#lemon pot
Lemon_pot_location = os.path.join(item_folder, "UmbrellaPot.stl")
Lemon_pot_scene = Mesh(filename=Lemon_pot_location, color="#504E4E", scale=[0.001, 0.001, 0.001])
Lemon_pot_scene.T = SE3(0.2, -0.2, 0.26)
env.add(Lemon_pot_scene)

#UR3 Base
ur3, ur3_base_scene = add_UR3(env)
set_UR3_pose(ur3, ur3_base_scene, SE3(0, 0.1, 0.15))
env.step()
input("Press Enter to continue...")
