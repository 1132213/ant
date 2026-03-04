from logic.map import MAP_SIZE

# map dimensions (hexagonal grid) now match ant_game deployment
col = MAP_SIZE
row = MAP_SIZE
# 将军升级所需金币
farmer_production_T1 = 10
farmer_production_T2 = 25
farmer_production_T3 = 35
farmer_defense_T1 = 10
farmer_defense_T2 = 15
farmer_defense_T3 = 30

lieutenant_new_recruit = 50
# tower upgrade costs (level1->2 and level2->3). according to rules each
# upgrade costs 60 then 200 gold regardless of tower type. earlier code used
# smaller constants; bump them here and production_up/defence_up logic will
# take effect. main generals still pay half of these values.
lieutenant_production_T1 = 60
lieutenant_production_T2 = 200
lieutenant_defense_T1 = 60
lieutenant_defense_T2 = 200

general_movement_T1 = 20
general_movement_T2 = 40
# 战法金币消耗
tactical_strike = 20
breakthrough = 15
leadership = 30
fortification = 30
weakening = 30

# 科技升级所需金币
army_movement_T1 = 80
army_movement_T2 = 150
mountaineering = 100
swamp_immunity = 75
unlock_super_weapon = 250

# base upgrade costs (production speed)
base_upgrade_speed_T1 = 200  # level1 -> level2
base_upgrade_speed_T2 = 250  # level2 -> level3

bog_percent = 0.15
mountain_percent = 0.05

farmer_num = 8
subgen_num = 4

sleep_time = 1
start_cd = 10
use_cd = 50
