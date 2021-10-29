# inputs
# construct a Json file within input data
""" product mix model optimization """
import json
# fixed inputs
frame_sales = {
    'unit_sell_price' : [28.5,12.5,29.25,21.50],
    'max_sales' : [2000,4000,1000,2000]
          }

# find the keys in data
frame_prod = {
    'labor' : {'hourly_wage' : 8.00,     'used_per_frame' : [2,1,3,2], 'res_available' : 4000},
    'metal' : {'cost_per_metal' : 0.50,  'used_per_frame' : [4,2,1,2], 'res_available' : 6000},
    'glass' : {'cost_per_glass' : 0.75,  'used_per_frame' : [4,2,1,2], 'res_available' : 6000}
}

print(frame_sales, type(frame_sales))
print(frame_prod, type(frame_prod))

# convert dictionary to JSON - use json.dumps()

file_name = 'inputs data for product mix model optimization.json'
with open(file_name, 'w') as file:
    json_string_1 = json.dumps(frame_sales, indent= 4)
    json_string_2 = json.dumps(frame_prod, indent= 4)
    file.write(json_string_1)
    file.write(json_string_2)
    file.close()

    
# how to solve json standard only allows one top level value
## combine as [{obj1], {obj2}] to use them in the same json file
