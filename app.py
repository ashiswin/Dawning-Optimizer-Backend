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


def solveILP(amounts, essence_cost):
    # Essence of Dawning guaranteed to be the last item in the list
    if amounts[-1] < essence_cost:
        return {"total": 0, "items": []}

    cost[-1] = np.ones(len(recipes)) * essence_cost
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
    return {"total": np.sum(x.value), "items": result}


@app.route("/calculate", methods=["POST"])
@cross_origin()
def calculate():
    amounts = []
    for ingredient in ingredients:
        if ingredient in request.json and request.json[ingredient] != "":
            amounts.append(request.json[ingredient])
        else:
            amounts.append(0)
    amounts = np.array(amounts, dtype=np.int64)

    # Calculate without masterwork
    response = solveILP(amounts, NOMASTERWORK_ESSENCE)

    # Calculate with masterwork
    mwresult = solveILP(amounts, MASTERWORKED_ESSENCE)
    response["mwtotal"] = mwresult["total"]
    response["mwitems"] = mwresult["items"]

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
