dogs = int(input("How many dogs do you have?\n",))
cats = int((input("How many cat do you have?\n",)))

if dogs == 1:
  total_dogs = f'{dogs} dog'
else:
  total_dogs = f'{dogs} dogs'

if cats == 1:
  total_cats = f'{cats} cat'
else:
  total_cats = f'{cats} cats'

print(f'You have {total_dogs} and {total_cats}')
