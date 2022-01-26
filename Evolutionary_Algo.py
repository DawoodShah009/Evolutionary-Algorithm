from math import log10
from typing import Sized
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
from numpy.lib.type_check import imag
from matplotlib.patches import Rectangle
import cv2
import random

large_image = cv2.imread("large pic.jpg",0)
small_image = cv2.imread("small pic.jpg",0)


def func_for_bit_count(size):
    no_of_bits = 0
    while size != 1 :
        size = size//2
        no_of_bits = no_of_bits + 1
    return no_of_bits


# reading images
size_of_large_pic = large_image.shape
size_of_small_pic = small_image.shape

# finding number of rows and columns of both images
no_of_rows_s = size_of_small_pic[0]
no_of_columns_s = size_of_small_pic[1]



no_of_rows_l = size_of_large_pic[0]
no_of_columns_l = size_of_large_pic[1]

#calculating large image  bits for x and y values;
row_bits = func_for_bit_count(no_of_rows_l)
column_bits = func_for_bit_count(no_of_columns_l)

best_fit_individual = []
best_fit_score = []
mean_fit_of_all_time = []
best_fit_of_all_time_dict = {}
population_size = 100

#===================================================================================================================
def Population_Initialization():

    #no_of_rows_l = number of rows in the large image(height of large image)
    #no_of_columns_l = number of columns in the large image(width of large image)
    #population_size was set to hundred
    population = [(random.randrange(0, no_of_rows_l), 
                  random.randrange(0, no_of_columns_l))for i in range(population_size)]
    #returning population of randomly generated 100 individuals
    return population

def correlation_coefficient(T1, T2): 
    numerator = np.mean((T1 - np.mean(T1)) * (T2 - np.mean(T2)))
    denominator = np.std(T1) * np.std(T2)
    if denominator == 0:
        return 0
    else:
        result = numerator / denominator
        return result

def Fitness_score(population):
    #large_image = reading the main image
    large_image = cv2.imread("large pic.jpg",0)
    #small_image = reading the template image
    small_image = cv2.imread("small pic.jpg",0)

    threshold = 0.85
    #no_of_rows_l = height of the main image
    no_of_rows_l = size_of_large_pic[0]
    #no_of_columns_l = width of the main image
    no_of_columns_l = size_of_large_pic[1]

    populationList = []
    global scoreList
    scoreList = []
    matrix_of_large_image = [[0 for i in range(29)] for j in range(35)]

    for point in population:
        row_value, column_value = point

        for r in range(no_of_rows_s): #no_of_rows_s = height of template image
            for c in range(no_of_columns_s): #no_of_columns_s = width of small image

                if (column_value + c) < no_of_columns_l and (row_value + r) < no_of_rows_l : #checking if any individual is near the 
        #             #boundry of the main image

                    matrix_of_large_image[r][c] = large_image[row_value+r][column_value+c]

        # correlation_coefficient= a sub function for calculating fitness score
        corelation = correlation_coefficient(matrix_of_large_image, small_image)

        scoreList.append(corelation)
        populationList.append(point)

        if corelation > threshold:
            point = populationList[len(populationList)-1]
            best_fit_individual.append(point)

            best_fit_score.append(corelation)
            mean_fit_value = np.mean(scoreList)
            mean_fit_of_all_time.append(mean_fit_value)
            print("The best point that has been discovered  is",point, corelation)
            display(point)
            return False


    scoreList.sort(reverse=True)

    best_fit_score.append(scoreList[0])

    populationList.sort(reverse=True)

    best_fit_individual.append(populationList[0])
 
    mean_fit_value = np.mean(scoreList)
    mean_fit_of_all_time.append(mean_fit_value)

    return populationList

def Sorting(populationList):
    zipped_lists = zip(scoreList, populationList) 
    sorted_zipped_lists = sorted(zipped_lists)
    Sorted_Population = [element for _, element in sorted_zipped_lists]
    
    return Sorted_Population

def bi_to_dec(ans):
    num = 0
    value_of_bit = 0
    for i in range(len(ans)) [::-1]:
        num = 2**value_of_bit * int(ans[i]) + num
        value_of_bit+=1
    return num

def func_for_merge(p1,p2,population_list):

    cross_over_point = random.randint(4,(len(p1)-4))
    crossed_parent1 = p1[0:cross_over_point] + p2[cross_over_point:len(p2)]
    crossed_parent2 = p2[0:cross_over_point] + p1[cross_over_point: len(p1)]

    num1 = bi_to_dec(crossed_parent1[0:9])
    num2 = bi_to_dec(crossed_parent1[9:len(crossed_parent1)])
    num3 = bi_to_dec(crossed_parent2[0:9])
    num4 = bi_to_dec(crossed_parent2[9:len(crossed_parent2)])

    off_spring1 = (num1, num2)
    off_spring2 = (num3, num4)

    population_list.append(off_spring1)
    population_list.append(off_spring2)
    return population_list

def convert_to_binary(value, check):
    list_of_bits = []
    if check == 0:
        size = row_bits
    else:
        size = column_bits

    for i in range(size):
        m = str((value%2))
        list_of_bits.append(m)
        value = value//2

    num = list_of_bits[len(list_of_bits)-1]

    for i in range(len(list_of_bits)-1)[::-1]:
        num = num + list_of_bits[i]
    return num

def cross_over(list):

    Population = []

    for i in range(0,(len(list)-1),2):


        # print(dict[i])
        p1,p2 = list[i]
        # print(dict[i+1])
        p3,p4 = list[i+1]

        row_val1 = convert_to_binary(p1,0)
        col_val1 = convert_to_binary(p2,1)
        parent1 = row_val1 + col_val1

        row_val2 = convert_to_binary(p3,0)
        col_val2 = convert_to_binary(p4,1)
        parent2 = row_val2 + col_val2

        Population = func_for_merge(parent1, parent2,Population)

    return Population

def stop_func(count, generation, first_max_fit_score, Population):
    if count <= 500:
        if first_max_fit_score == max(scoreList):
            count +=1
            first_max_fit_score = first_max_fit_score
        else:
            first_max_fit_score = max(scoreList)
            count = 0

    if count > 500:
        max_score = max(scoreList)

        key_list = list(best_fit_of_all_time_dict.keys())
        val_list = list(best_fit_of_all_time_dict.values())
        position = val_list.index(max_score)
        point = key_list[position]
        print("The best fit individual repeating for", count ,"generation is:",point, max_score)
        display(Population[0])
        return False

    if generation <= 10000:
        generation +=1
    else:
        sorted_population = sorted(best_fit_of_all_time_dict, key=best_fit_of_all_time_dict.get, reverse=True)
        point = sorted_population[0]
        print("The best fit  individual after",generation-1," generation is", point, best_fit_of_all_time_dict[point])
        display(point)
        return False

    return count, generation

def mutation(population):
    
    res = [random.randrange(1, 99) for i in range(3)]

    for i in res: 
        num1, num2 = population[i]
        num1 = convert_to_binary(num1,0)
        num2 = convert_to_binary(num2,1)

        tuple1 = (num1, num2)
        # print(i)
        # bit_index_row = random.randint(0,18)
        bit_index_row = random.randint(0,8)


        bit_index_column = random.randint(0,9)

        num1 = tuple1[0]
        num2 = tuple1[1]

        l1 = list(num1)
        l2 = list(num2)

        if l1[bit_index_row] == "0":
            l1[bit_index_row] = "1"
        else:
            l1[bit_index_row] = "0"

        if l2[bit_index_column] == "0":
                l2[bit_index_column] = "1"
        else:
                l2[bit_index_column] = "0"
        num1 = "".join(l1)
        num2 = "".join(l2)

        num1 = bi_to_dec(num1)
        num2 = bi_to_dec(num2)
        new_individual = (num1,num2)
        population[i] = new_individual
    return population

def display(point):
    window_name = 'Image'

    w = point[1]
    h = point[0]

    start_point = (w, h)
    end_point = (w+no_of_columns_s, h+no_of_rows_s)
    color = (255, 0, 0)
    thickness = 1
    image = cv2.rectangle(large_image, start_point, end_point,color, thickness)

    cv2.imshow(window_name, image) 
    cv2.waitKey(0) 

def show_grapgh():
    plt.plot( mean_fit_of_all_time,label = "mean", )
    plt.plot(best_fit_score, label = "best fit")

    plt.legend()
    plt.show()

#=================================================================================================
def start():

    Population = Population_Initialization()
    Population= Fitness_score(Population)
    # print(Population)

    if Population == False:
        StopIteration
    else:

        # value_list = list(Population.values())
        first_max_fit_score = max(scoreList)

        max_fit_count = 1
        max_generation = 1

        while stop_func(max_fit_count, max_generation, first_max_fit_score,Population) != False:

            max_fit_count,max_generation =  stop_func(max_fit_count, max_generation, first_max_fit_score,Population)
            Population = Sorting(Population)
            Population = cross_over(Population)
            # print(Population)
            Population = mutation(Population)
            # print(Population)
            Population= Fitness_score(Population)
            # print(Population)
            if Population == False:
                break
    show_grapgh()


# Uncoment start() to start the program:
# start()

#==================================================================================================








