import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
import numpy as np
import cvxpy as cp

MASTERWORKED_ESSENCE = 10
NOMASTERWORK_ESSENCE = 15

app = Flask(__name__, static_url_path="")
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

ingredients = []

with open("ingredients.csv") as f:
    for i in f:
        ingredients.append(i.strip())

recipes = []
cost = None

with open("cost.csv") as f:
    recipes = f.readline().split(",")

    cost = np.zeros((len(ingredients), len(recipes)))

    i = 0
    for line in f:
        row = line.split(",")

        for j in range(len(row)):
            cost[i][j] = int(row[j])

        i += 1

def parseConstraints(x, constraints):
    constraintList = []

    for constraint in constraints:
        name = constraint["name"]
        equality = constraint["equality"]
        value = constraint["value"]

        index = recipes.index(name)

        if equality == "lte":
            constraintList.append(x[index] <= value)
        elif equality == "eq":
            constraintList.append(x[index] == value)
        elif equality == "gte":
            constraintList.append(x[index] >= value)
        else:
            continue
    return constraintList

def solveILP(amounts, essence_cost, rawConstraints):
    # Essence of Dawning guaranteed to be the last item in the list
    if amounts[-1] < essence_cost:
        return {"total": 0, "items": []}

    cost[-1] = np.ones(len(recipes)) * essence_cost
    x = cp.Variable(len(recipes), integer=True)

    objective = cp.Maximize(cp.sum(cost * x))
    constraints = [cost * x <= amounts, x >= 0]
    constraints.extend(parseConstraints(x, rawConstraints))
    prob = cp.Problem(objective, constraints)

    prob.solve(solver=cp.GLPK_MI)

    if x.value is None:
        return {"total": 0, "items": []}

    result = []

    for i in range(x.value.shape[0]):
        if x.value[i] > 0:
            result.append((recipes[i], x.value[i]))
    result.sort(key=lambda x: -x[1])
    return {"total": np.sum(x.value), "items": result}


@app.route("/calculate", methods=["POST"])
@cross_origin()
def calculate():
    quantities = request.json["quantities"]
    constraints = request.json["constraints"]

    amounts = []
    for ingredient in ingredients:
        if ingredient in quantities and quantities[ingredient] != "":
            amounts.append(quantities[ingredient])
        else:
            amounts.append(0)
    amounts = np.array(amounts, dtype=np.int64)

    # Calculate without masterwork
    response = solveILP(amounts, NOMASTERWORK_ESSENCE, constraints)

    # Calculate with masterwork
    mwresult = solveILP(amounts, MASTERWORKED_ESSENCE, constraints)
    response["mwtotal"] = mwresult["total"]
    response["mwitems"] = mwresult["items"]

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
