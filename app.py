import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cvxpy as cp

app = Flask(__name__, static_url_path='')
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

ingredients = []

with open('ingredients.csv') as f:
    for i in f:
        ingredients.append(i.strip())

recipes = []
cost = None

with open('cost.csv') as f:
    recipes = f.readline().split(',')

    cost = np.zeros((len(ingredients), len(recipes)))

    i = 0
    for line in f:
        row = line.split(',')

        for j in range(len(row)):
            cost[i][j] = int(row[j])
        
        i += 1
        
@app.route('/calculate', methods=['POST'])
def calculate():
    amounts = []
    for ingredient in ingredients:
        if ingredient in request.json and request.json[ingredient] != "":
            amounts.append(request.json[ingredient])
        else:
            amounts.append(0)
    amounts = np.array(amounts, dtype=np.int64)

    # Calculate without masterwork
    cost[-1] = np.ones(len(recipes)) * 15
    x = cp.Variable(len(recipes), integer=True)

    objective = cp.Maximize(cp.sum(cost * x))
    constraints = [cost * x <= amounts, x >= 0]
    prob = cp.Problem(objective, constraints)

    prob.solve(solver=cp.GLPK_MI)

    result = []

    for i in range(x.value.shape[0]):
        if x.value[i] > 0:
            result.append((recipes[i], x.value[i]))
    result.sort(key=lambda x: -x[1])
    response = {'total': np.sum(x.value), 'items': result}

    # Calculate with masterwork
    cost[-1] = np.ones(len(recipes)) * 10

    prob.solve(solver=cp.GLPK_MI)
    mwresult = []

    for i in range(x.value.shape[0]):
        if x.value[i] > 0:
            mwresult.append((recipes[i], x.value[i]))
    mwresult.sort(key=lambda x: -x[1])
    response['mwtotal'] = np.sum(x.value)
    response['mwitems'] = mwresult

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)