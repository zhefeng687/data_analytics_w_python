"""
Title: Product mix optimization problem

## stage problem

- ** $Supplies Cost of Frame 1, 2, 3, 4
- ** # of Produced Frame 1, 2, 3, 4
- ** $Product Cost of Frame 1, 2, 3, 4
- ** $Total Cost of Frame
    - $(supplies cost + product cost ) * # of produced Frame
- ** #Total Resource Used
    - (# of Produced Frame 1, 2, 3, 4 ) * ( # of resource used per Frame)
- ** # of Max Sales of Frame 1, 2, 3, 4
- ** $Revenue
    - $Unit selling price * # of produced Frame
- ** Decision variables **
    - # of Produced Frame 1, 2, 3, 4
- ** Objective **
    - $Max_Profit \ $Revenue - $Total Cost of Frame
    - lpSum() – given a list of the form [a1*x1, a2x2, …, anxn] will construct a linear expression to be used as a constraint or variable
- ** Supplies Constraints **
    - # of resources used <= resource available
- ** Production Constraints**
    - # of Produced Frame 1, 2, 3, 4 <= # of Max Sales of Frame 1, 2, 3, 4
- ** Non-negativity **

"""

from pulp import *

frame_sales = {
    'unit_sell_price' : [28.5,12.5,29.25,21.50],
    'max_sales' : [2000,4000,1000,2000]
          }

# find the keys in data
frame_prod = {
    'labor' : {'hourly_wage' :    8.00,  'used_per_frame' : [2,1,3,2], 'res_available' : 4000},
    'metal' : {'cost_per_metal' : 0.50,  'used_per_frame' : [4,2,1,2], 'res_available' : 6000},
    'glass' : {'cost_per_glass' : 0.75,  'used_per_frame' : [4,2,1,2], 'res_available' : 6000}
}

print(frame_sales, type(frame_sales))
print(frame_prod, type(frame_prod))

# define the optimization problem
product_mix_model = LpProblem('Product_mix_optimization_model', LpMaximize)

# create referenced variables
prodList=[LpVariable(f'frame_{x+1}', 0, frame_sales['max_sales'][x], LpInteger) for x in range(len(frame_sales['max_sales']))]

# set Objective - prob+=
# revenue
# revenue = [frame_sales['unit_sell_price'][x] * prodList[x] for x in range(len(prodList))]
# total_labor_cost + total resource cost
# total_cost = ([prodList[x] * (frame_prod['labor']['hourly_wage'] * frame_prod['labor']['used_per_frame'][x]
#             + frame_prod['metal']['cost_per_metal']* frame_prod['metal']['used_per_frame'][x]
#             + frame_prod['glass']['cost_per_glass']* frame_prod['metal']['used_per_frame'][x])
#               for x in range(len(prodList))])
product_mix_model += lpSum([frame_sales['unit_sell_price'][x] * prodList[x]] - prodList[x]
                            * (frame_prod['labor']['hourly_wage'] * frame_prod['labor']['used_per_frame'][x]
                            + frame_prod['metal']['cost_per_metal']* frame_prod['metal']['used_per_frame'][x]
                            + frame_prod['glass']['cost_per_glass']* frame_prod['metal']['used_per_frame'][x])
                             for x in range(len(prodList)))

# resource constraint (- prob+=)


for i in frame_prod.keys():
    product_mix_model += lpSum([prodList[x] * frame_prod[i]['used_per_frame'][x] for x in range(len(prodList))]) <= frame_prod[i]['res_available']




# The problem is solved using PuLP's choice of Solver
product_mix_model.solve()

# output the optimization
print('status:', LpStatus[product_mix_model.status])

# Each of the variables is printed with it's resolved optimum value
print('The products production plan are as follows for the max profits:')
for v in product_mix_model.variables():
   print( v.name, "=", v.value())

# The optimised objective function value
print("Revenue = ", value(product_mix_model.objective))


# Or list comprehension
# prod = [(v.name, v.value()) for v in product_mix_model.variables()]
# print(prod)
