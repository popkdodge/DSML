''' people = 20
cats = 30
dogs = 15

if people < cats:
  print ("Too many cats! The world is doom")
if people > cats:
  print ("Not many cats! The world is saved !")
if people < dogs:
  print("The world is drooled on!")
if people > dogs:
  print("The world is dry!")

dogs += 5

if people >= dogs:
  print('People are greater than or equal to dogs.')
if people <= dogs:
  print("People are less than or equal to dogs.")

if people == dogs:
  print("People are dogs.")  '''
'''
people = 30
cars = 40
buses = 15

if cars > people:
  print ("We should take the cars.")
elif  cars < people:
  print (" We should not take the cars.")

if buses > cars:
  print ("That's too many busses.")
elif buses < cars:
  print ("Maybe we could take the buses.")
else:
  print ("We still can't decide.")

if people > buses:
  print ("Alright, let's just take the buses.")
else:
  print ("Fine, let's stay home then.")'''
'''
print ("You enter a dark room with two doors. Do you go through door #1 or door #2?")
door = input()

if door == "1":
  print ("There a giant bear here eating a cheese cake.  What do you do?")
  print ("1. Take the cake.")
  print ("2. Scream at the bear.")

  bear = input()

  if bear == '1':
    print ("The bear eats your face off.  Good job!")
  elif bear == '2':
    print ("The bear eats your legs off.  Good job!")
  else:
    print ("Well you, doing %s is probably better. Bear runs away.") %bear
elif door == '2':
  print ("You stear in to the endless abyss at Cthulhu's retna.")
  print ("1. Blueberries.")
  print ("2. Yellow yacket clothespins.")
  print ("3. Understanding revolvers yelling melodies.")
  
  insanity = input()

  if insanity == '1' or insanity == '2':
    print ('Your body survives powered by a mind of jello.  Good job!')
  else:
    print ("The insanity rots your eyes into a pool of muck.  Good job!")
else:
  print ("You stumble around and fall on a knife and die. Good job!")
  '''

count_count = [1, 2, 3, 4, 5]
fruits = ['apples', 'oranges', 'pears', 'apricots']
changes = [1, 'pennies', 2, 'dimes', 3]

for number in count_count:
  print('This is count %d') %number

for fruit in fruits:
  print('A fruit of type: %s' % fruit)

