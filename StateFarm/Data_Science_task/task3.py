dinosaur_names = ["triceratops", "t-rex", "velociraptor", "altascopcosauras"]

print("Welcome to the big dino-name program!")
big_name = int(input('Enter a name length you consider to be really big:\n'))

for name in dinosaur_names:
  if len(name) > big_name:
    print(f'{name} is really big')
