#!/usr/bin/python3
import math

print("give me a bottle of rum!")
x = 4
print(math.sqrt(x))      # sqrt(4) = 2
print(math.pow(x,2))     # 4**2 = 16
print(math.exp(x))       # exp(4) = 54.6
print(math.log(x,2))     # log based 2  (default is natural logarithm)
print(math.fabs(-4))     # absolute value
print(math.factorial(x)) # 4! = 4 x 3 x 2 x 1 = 24

z = 0.2
print(math.ceil(z))      # ceiling function
print(math.floor(z))     # floor function
print(math.trunc(z))     # truncate function

z = 3*math.pi            # math.pi = 3.141592653589793 
print(math.sin(z))       # sine function
print(math.tanh(z))      # arctan function

x = math.nan             # not a number
print(math.isnan(x))

x = math.inf             # infinity
print(math.isinf(x))


separator = " "
print(separator.join(mylist))    # merge all elements of the list into a string

keys = ['apples', 'oranges', 'bananas', 'cherries']
values = [3, 4, 2, 10]
fruits = dict(zip(keys, values))
print(fruits)
print(sorted(fruits))     # sort keys of dictionary

mylist = ['this', 'is', 'a', 'list']
for word in mylist:
    print(word.replace("is", "at"))

mylist2 = [len(word) for word in mylist]   # number of characters in each word
print(mylist2)

states = [('MI', 'Michigan', 'Lansing'),('CA', 'California', 'Sacramento'),
          ('TX', 'Texas', 'Austin')]

sorted_capitals = [state[2] for state in states]
sorted_capitals.sort()
print(sorted_capitals)

fruits = {'apples': 3, 'oranges': 4, 'bananas': 2, 'cherries': 10}
fruitnames = [k for (k,v) in fruits.items()]
print(fruitnames)

myfunc = lambda x: 3*x**2 - 2*x + 3      # example of an unnamed quadratic function

print(myfunc(2))

with open('states.txt', 'r') as f:
    for line in f:
        fields = line.split(sep=',')    # split each line into its respective fields
        print('State=',fields[1],'(',fields[0],')','Capital:', fields[2])

#
#PWN
#
hex()
text="assf"
#取第四行字符串，分割取最后一项，按十六进制表示
int(text.splitlines()[4].split()[-1], 16)