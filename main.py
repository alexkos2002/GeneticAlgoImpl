from random import randint
from random import sample
from random import choices
from random import choice
from math import ceil
from math import floor
import tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from math import sqrt
from copy import deepcopy
import numpy as np

geneticAlgoSolution = None
CANVAS_WIDTH = 700
CANVAS_HEIGHT = 850
EMPTY_STRING = ""

def startGA():
    geneticAlgoSolution = GeneticAlgoSalesmanProblemSolution(int(iter_num_input.get()), int(cities_num_input.get()),
                                                             int(population_size_input.get()), float(mutation_percentage_input.get()),
                                                             float(max_sat_distance_input.get()), float(min_convergence_dist_delta_input.get()),
                                                             int(convergence_iters_input.get()))
    generation_winners = geneticAlgoSolution.start()
    print(generation_winners)
    generation_winners_plot_x_y_list = geneticAlgoSolution.convert_generation_winners_to_plot_lists(generation_winners)

    for i in range(0, len(generation_winners_plot_x_y_list), 2):
        t = np.arange(0, 3, .01)
        if i == len(generation_winners_plot_x_y_list) - 2:
            ax.plot(generation_winners_plot_x_y_list[i], generation_winners_plot_x_y_list[i + 1], linewidth=3.0)
        else:
            ax.plot(generation_winners_plot_x_y_list[i], generation_winners_plot_x_y_list[i + 1])

    canvas.draw()
    canvas.get_tk_widget().pack(side=tkinter.BOTTOM, fill=tkinter.BOTH, expand=1)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tkinter.BOTTOM, fill=tkinter.BOTH, expand=1)

def clear_figure():
    plt.clf()
    fig.add_subplot(111)
    ax = fig.get_axes()[0]
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tkinter.BOTTOM, fill=tkinter.BOTH, expand=1)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tkinter.BOTTOM, fill=tkinter.BOTH, expand=1)


def quit():
    root.quit()
    root.destroy()

class GeneticAlgoSalesmanProblemSolution:

    def __init__(self, iter_num, cities_num, population_size, mutation_percentage, max_sat_distance, min_convergence_dist_delta, convergence_iters):
        self.iter_num = iter_num
        self.cities_num = cities_num
        self.population_size = population_size
        self.mutation_percentage = mutation_percentage
        self.max_sat_distance = max_sat_distance
        self.min_convergence_dist_delta = min_convergence_dist_delta
        self.convergence_iters = convergence_iters
        self.cities = None
        self.gen_len = len(bin(cities_num - 1)[2:])
        self.genome_len = self.gen_len * cities_num
        self.individuals = None
        self.generation_winners = []

    def start(self):
        self.generate_cities()
        self.generate_individuals()
        count_iter = 0
        convergence_delta = float('inf')
        min_sat_score = self.calculate_score_by_distance(self.max_sat_distance)
        min_convergrnce_delta = self.calculate_score_by_distance(self.min_convergence_dist_delta)
        cur_generation_winner = {
            'genome': None,
            'score': 0.0
        }

        while not (count_iter >= self.iter_num or convergence_delta < min_convergrnce_delta and
                   count_iter >= self.convergence_iters - 1 or cur_generation_winner['score'] >= min_sat_score):
            print("Generation number: ", count_iter)
            print(count_iter, "Generation individuals: ", self.individuals)
            self.calculate_scores_for_individuals()
            cur_generation_winner = self.find_generation_winner()
            print(count_iter, "Generation winner: ", cur_generation_winner)
            self.generation_winners.append(cur_generation_winner)
            next_gen_individuals = []
            next_gen_parents = self.select_parents_by_best_scores()
            next_gen_parents = self.retrieve_binary_genomes_from_individuals(next_gen_parents)
            next_gen_parents_pairs = self.group_parents_into_pairs(deepcopy(next_gen_parents))
            for firstParentInd in range(0, len(next_gen_parents_pairs), 2):
                next_gen_individuals.extend(self.execute_multi_point_crossover(next_gen_parents_pairs[firstParentInd],
                                                                               next_gen_parents_pairs[firstParentInd + 1]))
            mutated_next_gen_individuals = self.execute_mutation(next_gen_individuals)
            validated_individuals = self.validate_next_gen_individuals(mutated_next_gen_individuals, next_gen_parents)
            self.individuals = self.convert_binary_genomes_to_individuals(validated_individuals)
            if count_iter >= self.convergence_iters:
                convergence_delta = cur_generation_winner['score'] - self.generation_winners[count_iter - self.convergence_iters]['score']
            count_iter += 1
        print("Generation winners' scores", [winner['score'] for winner in self.generation_winners])
        print("Generation winners' distances", [self.calculate_distance_by_score(winner['score']) for winner in self.generation_winners])
        return self.generation_winners

    def generate_cities(self):
        self.cities = [
            {
                'id': i,
                'pos': (
                    randint(0, CANVAS_WIDTH),
                    randint(0, CANVAS_HEIGHT))
            } for i in range(self.cities_num)
        ]
        print("Citiss: ", self.cities)

    def generate_individuals(self):
        self.individuals = [
            {
                'genome': sample(self.cities, self.cities_num),
                'score': 0
            } for i in range(self.population_size)
        ]

    def convert_city_id_gen_to_binary_gen(self, city_id_gen):
        return ((bin(city_id_gen))[2:]).rjust(self.gen_len, '0')

    def retrieve_binary_genomes_from_individuals(self, individuals):
        binary_genomes_genes = [
            [self.convert_city_id_gen_to_binary_gen(gen['id']) for gen in genome] for genome in [ind['genome'] for ind in individuals]
        ]
        binary_genomes = []
        for bg in binary_genomes_genes:
            binary_genomes.append(EMPTY_STRING.join(bg))
        return binary_genomes

    def convert_binary_genomes_to_individuals(self, binary_genomes):
        city_id_genomes = self.convert_binary_genomes_to_city_id_genomes(binary_genomes)
        genomes = [[list(filter(lambda city: city['id'] == city_id, self.cities)).pop(0) for city_id in cng] for cng in city_id_genomes]
        return [
            {
                'genome': genome,
                'score': 0,
            } for genome in genomes
        ]

    def convert_binary_genome_to_city_id_genome(self, binary_genome):
        return [int(binary_genome[i: i + self.gen_len], 2) if int(binary_genome[i: i + self.gen_len], 2) < self.cities_num else int(binary_genome[i: i + self.gen_len], 2) % self.cities_num for i in range(0, len(binary_genome), self.gen_len)]

    def convert_binary_genomes_to_city_id_genomes(self, binary_genomes):
        return [self.convert_binary_genome_to_city_id_genome(genome) for genome in binary_genomes]

    def calculate_distance_between_points(self, p1, p2):
        return sqrt(
            (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
        )

    def calculate_score_by_distance(self, distance):
        return sqrt(CANVAS_WIDTH * CANVAS_HEIGHT) / distance

    def calculate_scores_for_individuals(self):
        ind_distance = 0
        for ind in self.individuals:
            ind_genome = ind['genome']
            for city_index in range(len(ind_genome) - 1):
                ind_distance += self.calculate_distance_between_points(ind_genome[city_index]['pos'], ind_genome[city_index + 1]['pos'])
            ind['score'] = round(self.calculate_score_by_distance(ind_distance), 5)
            ind_distance = 0

    def find_generation_winner(self):
        return max(self.individuals, key=lambda ind: ind['score'])

    def select_parents_by_best_scores(self):
        ind_indexes_to_scores = {i: ind['score'] for i, ind in enumerate(self.individuals)}
        ind_indexes = list(ind_indexes_to_scores.keys())
        ind_scores = ind_indexes_to_scores.values()
        min_score = min(ind_scores)
        ind_scores = [ind_score - min_score for ind_score in ind_scores]
        scores_sum = sum(ind_scores)
        ind_probabilities = [round(score / scores_sum, 5) for score in ind_scores]
        parents_ids = choices(ind_indexes, ind_probabilities, k=len(ind_indexes))
        parents = [self.individuals[index] for index in parents_ids]
        return parents

    def select_2_halves_parents_by_best_scores(self):
        ind_indexes_to_scores = {i: ind['score'] for i, ind in enumerate(self.individuals)}
        sorted_ind_indexes_to_scores = dict(sorted(ind_indexes_to_scores.items(), key=lambda x: x[1], reverse=True))
        ind_indexes = list(ind_indexes_to_scores.keys())
        ind_scores = list(ind_indexes_to_scores.values())
        scores_sum = sum(ind_scores)
        ind_probabilities = [round(score / scores_sum, 5) for score in ind_scores[:ceil(len(ind_indexes) / 2)]]
        parents_ids = choices(ind_indexes[:ceil(len(ind_indexes) / 2)], ind_probabilities, k=len(ind_indexes))
        parents = [self.individuals[index] for index in parents_ids]
        return parents

    def calculate_hamming_distance(self, binary_genome1, binary_genome2):
        return sum([int(gen1 == gen2) for gen1, gen2 in zip(binary_genome1, binary_genome2)])

    def group_parents_into_pairs(self, parents_binary_genomes):
        first_parent_index, second_parent_index = 0, 0
        parents_pairs = []
        while len(parents_binary_genomes) != 0:
            first_parent_index = choice(range(len(parents_binary_genomes)))
            first_parent = parents_binary_genomes.pop(first_parent_index)
            second_parent_index = parents_binary_genomes.index(max(parents_binary_genomes, key=lambda p: self.calculate_hamming_distance(first_parent, p)))
            second_parent = parents_binary_genomes.pop(second_parent_index)
            parents_pairs.extend([first_parent, second_parent])
        return parents_pairs

    def get_city_id_genome_duplicate_genes_amounts(self, city_id_genome):
        duplicate_genes_amounts = {city_id : 0 for city_id in city_id_genome}
        sorted_city_id_genome = sorted(city_id_genome)
        prev_city_id_distinct_gen = -1
        for city_id_gen in sorted_city_id_genome:
            if prev_city_id_distinct_gen == city_id_gen:
                duplicate_genes_amounts[city_id_gen] = duplicate_genes_amounts[city_id_gen] + 1
            prev_city_id_distinct_gen = city_id_gen
        return dict(filter(lambda elem: elem[1] > 0, duplicate_genes_amounts.items()))

    #This function checks if all cities are present only once in individual's genome(route)
    def validate_binary_genome(self, binary_genome):
        binary_genome_genes = [binary_genome[i: i + self.gen_len] for i in range(0, len(binary_genome), self.gen_len)]
        sorted_binary_genome_genes = sorted(binary_genome_genes)
        prev_distinct_gen = ''
        for gen in sorted_binary_genome_genes:
            if prev_distinct_gen == gen:
                return False
                print("Binary genome validation failed")
            prev_distinct_gen = gen
        return True

    #if individual isn't valid(some citites are present more than one time in the route) after crossover and mutation
    #then its duplicated genes(cities) are replaced by the closest by Hamming distance remaining not present in genome
    #gen(city)
    def validate_next_gen_individuals(self, individuals, their_parents):
        for i in range(len(individuals)):
            cur_individual = individuals[i]
            cur_individual_city_id_genome = self.convert_binary_genome_to_city_id_genome(cur_individual)
            if not self.validate_binary_genome(cur_individual):
                city_id_genome_duplicate_genes_amounts = self.get_city_id_genome_duplicate_genes_amounts(cur_individual_city_id_genome)
                duplicate_city_id_genes = city_id_genome_duplicate_genes_amounts.keys()
                not_used_city_id_genes = [gen for gen in list(range(self.cities_num)) if gen not in cur_individual_city_id_genome]
                not_used_binary_genes = [self.convert_city_id_gen_to_binary_gen(city_id_gen) for city_id_gen in not_used_city_id_genes]
                for duplicate_city_id_gen in duplicate_city_id_genes:
                    for j in range(city_id_genome_duplicate_genes_amounts[duplicate_city_id_gen]):
                        duplicate_gen_index = cur_individual_city_id_genome.index(duplicate_city_id_gen)
                        binary_duplicate_city_id_gen = self.convert_city_id_gen_to_binary_gen(duplicate_city_id_gen)
                        closest_to_duplicate_gen_index = not_used_binary_genes.index(max(not_used_binary_genes, key=lambda g: self.calculate_hamming_distance(binary_duplicate_city_id_gen, g)))
                        closest_to_duplicate_gen = not_used_binary_genes.pop(closest_to_duplicate_gen_index)
                        cur_individual = cur_individual[:(duplicate_gen_index * self.gen_len)] + closest_to_duplicate_gen + cur_individual[(duplicate_gen_index * self.gen_len + self.gen_len):]
                individuals[i] = cur_individual
        return individuals

    #in this case double point random crossover
    def execute_multi_point_crossover(self, parent_genome1, parent_genome2):
        genome_len = len(parent_genome1)
        gen_len = self.gen_len
        parent_gen_list1 = list(parent_genome1)
        parent_gen_list2 = list(parent_genome2)
        if genome_len == gen_len:
            return parent_genome1, parent_genome2
        if genome_len == 2 * gen_len:
            parent_gen_list1[gen_len:], parent_gen_list2[gen_len:] = parent_gen_list2[gen_len:], parent_gen_list1[gen_len:]
            return EMPTY_STRING.join(parent_gen_list1), EMPTY_STRING.join(parent_gen_list2)
        else:
            slice1_start_index = choice(range(0, genome_len - 2 * gen_len, gen_len))
            slice1_end_index = choice(range(slice1_start_index + gen_len, genome_len - gen_len, gen_len))
            slice2_start_index = choice(list(range(0, slice1_start_index - gen_len, gen_len)) + list(range(slice1_end_index + gen_len, genome_len, gen_len))
                                        if slice1_start_index > gen_len else range(slice1_end_index + gen_len, genome_len, gen_len))
            slice2_end_index = choice(range(slice2_start_index + gen_len, slice1_start_index, gen_len)
                                      if slice1_start_index > slice2_start_index else range(slice2_start_index + gen_len, genome_len + 1, gen_len))
            parent_gen_list1[slice1_start_index:slice1_end_index], parent_gen_list2[slice1_start_index:slice1_end_index] =\
                parent_gen_list2[slice1_start_index:slice1_end_index], parent_gen_list1[slice1_start_index:slice1_end_index]
            parent_gen_list1[slice2_start_index:slice2_end_index], parent_gen_list2[slice2_start_index:slice2_end_index] =\
                parent_gen_list2[slice2_start_index:slice2_end_index], parent_gen_list1[slice2_start_index:slice2_end_index]
            return EMPTY_STRING.join(parent_gen_list1), EMPTY_STRING.join(parent_gen_list2)

    def execute_mutation(self, individuals_genomes):
        individuals_genomes_binary_string = EMPTY_STRING.join(individuals_genomes)
        individuals_genomes_binary_string_len = len(individuals_genomes_binary_string)
        mutations_number = floor(individuals_genomes_binary_string_len * self.mutation_percentage / 100) if self.mutation_percentage > 0 else 0
        if mutations_number == 0:
            return individuals_genomes
        individuals_genomes_binary_list = list(individuals_genomes_binary_string)
        print("Mutations number: ", mutations_number)
        mutations_indexes = sample(range(individuals_genomes_binary_string_len), mutations_number)
        for i in range(individuals_genomes_binary_string_len):
            if i in mutations_indexes:
                individuals_genomes_binary_list[i] = ('1' if individuals_genomes_binary_string[i] == '0' else '0')
        cur_individual_genome = ""
        gen_counter = 0
        mutated_individuals_genomes = []
        for gen in individuals_genomes_binary_list:
            cur_individual_genome += gen
            gen_counter += 1
            if gen_counter == self.genome_len:
                mutated_individuals_genomes.append(cur_individual_genome)
                cur_individual_genome = ""
                gen_counter = 0
        return mutated_individuals_genomes

    def convert_generation_winners_to_plot_lists(self, generation_winners):
        generation_winner_plot_lists = []
        cur_x_plot_list = []
        cur_y_plot_list = []
        for generation_winner in generation_winners:
            for city_gen in generation_winner['genome']:
                cur_x_plot_list.append((city_gen['pos'])[0])
                cur_y_plot_list.append((city_gen['pos'])[1])
            generation_winner_plot_lists.append(cur_x_plot_list)
            generation_winner_plot_lists.append(cur_y_plot_list)
            cur_x_plot_list = []
            cur_y_plot_list = []
        return generation_winner_plot_lists

    def calculate_distance_by_score(self, score):
        return sqrt(CANVAS_WIDTH * CANVAS_HEIGHT) / score

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    root = tkinter.Tk()
    root.title("Genetic Algorithm Salesman Problem")
    root.geometry(str(CANVAS_WIDTH) + "x" + str(CANVAS_HEIGHT))

    fig = plt.Figure(figsize=(5,4), dpi=100)

    fig.add_subplot(111)
    ax = fig.get_axes()[0]

    canvas = FigureCanvasTkAgg(fig, master=root)
    toolbar = NavigationToolbar2Tk(canvas, root)

    iter_num_lbl = tkinter.Label(root, text="Iterations number")
    iter_num_input = tkinter.Entry(root)

    cities_num_lbl = tkinter.Label(root, text="Cities number")
    cities_num_input = tkinter.Entry(root)

    population_size_lbl = tkinter.Label(root, text="Population size")
    population_size_input = tkinter.Entry(root)

    mutation_percentage_lbl = tkinter.Label(root, text="Mutation percentage")
    mutation_percentage_input = tkinter.Entry(root)

    max_sat_distance_lbl = tkinter.Label(root, text="Maximum satisfied distance of the route")
    max_sat_distance_input = tkinter.Entry(root)

    min_convergence_dist_delta_lbl = tkinter.Label(root, text="Minimal distance improvement")
    min_convergence_dist_delta_input = tkinter.Entry(root)

    convergence_iters_lbl = tkinter.Label(root, text="Number of iterations to reach minimal improvement")
    convergence_iters_input = tkinter.Entry(root)

    start_button = tkinter.Button(root, text="Start", command=startGA)
    clear_button = tkinter.Button(root, text="Clear", command=clear_figure)
    quit_button = tkinter.Button(root, text="Quit", command=quit)

    iter_num_lbl.pack()
    iter_num_input.pack()
    cities_num_lbl.pack()
    cities_num_input.pack()
    population_size_lbl.pack()
    population_size_input.pack()
    mutation_percentage_lbl.pack()
    mutation_percentage_input.pack()
    max_sat_distance_lbl.pack()
    max_sat_distance_input.pack()
    min_convergence_dist_delta_lbl.pack()
    min_convergence_dist_delta_input.pack()
    convergence_iters_lbl.pack()
    convergence_iters_input.pack()
    start_button.pack(pady=10)
    clear_button.pack(pady=10)
    quit_button.pack(pady=10)
    root.mainloop()
