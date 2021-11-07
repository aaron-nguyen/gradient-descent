import numpy as np
import random
import matplotlib.pyplot as plt

class AirportProblem:

    # Area of interest N x N km2
    N = 10000

    def __init__(self, number_of_airport, number_of_cities=10):
        super().__init__()
        self.n_cities = number_of_cities
        self.n_airport = number_of_airport
        self.city_location = self.N * np.random.random_sample((self.n_cities, 2))
        self.airport_location = self.N * np.random.random_sample((self.n_airport, 2))
        self.learning_rate = 0.005

    def costFunction(self):
        cost = 0
        for air in range(self.n_airport):
            singlecost = 0
            for city in range(self.n_cities):
                singlecost += (self.airport_location[air][0] - self.city_location[city][0]) ** 2 + (self.airport_location[air][0] - self.city_location[city][0]) ** 2

            cost += singlecost

        return cost    

    def firstDerivative(self):
        # (x - a)^2  --> 2(x - a)x = 2(x^2 - ax)
        df = np.zeros((self.n_airport*2, 1))

        for i in range(self.n_airport):
            for city in range(self.n_cities):
                df[2 * i] += 2 * (self.airport_location[i][0] - self.city_location[city][0]) * self.airport_location[i][0]
                df[2 * i + 1] += 2 * (self.airport_location[i][1] - self.city_location[city][1]) * self.airport_location[i][1]
            
        return df

    def hessianMatrix(self):
        # (x - a)^2  --> 2(x - a)x = 2(x^2 - ax) --> 4x - 2a
        d2f = np.zeros((self.n_airport*2, 1))

        for i in range(self.n_airport):
            for city in range(self.n_cities):
                d2f[2 * i] += 4 * self.airport_location[i][0] - 2 * self.city_location[city][0]
                d2f[2 * i + 1] += 4 * self.airport_location[i][1] - 2 * self.city_location[city][1]
        
        hessian = np.identity(self.n_airport * 2)
        hessian = hessian * (1 / d2f)

        return hessian

    def update(self):
        flatten_airport_location = self.airport_location.reshape(-1,1)
        df = self.firstDerivative()
        hessian = self.hessianMatrix()

        # update 
        flatten_airport_location = flatten_airport_location - self.learning_rate * np.matmul(hessian, df)
        self.airport_location = flatten_airport_location.reshape(-1, 2)


    def train(self, epochs):
        
        print(f"Initial airports' locations: \n{self.airport_location}\n")

        costs = []

        for epoch in range(epochs):
            cost = self.costFunction()
            self.update()
            costs.append(cost)

        print(f"Final airports' locations: \n{self.airport_location}\n")

        plt.plot(costs)
        plt.savefig("cost.png")

'''
TEST
'''

def main():
    problem = AirportProblem(number_of_airport=3)

    print(problem.firstDerivative())

    print(problem.hessianMatrix())

    problem.train(epochs= 100)


if __name__ == "__main__":
    main()