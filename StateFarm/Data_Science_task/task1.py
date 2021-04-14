# Task 1

age = 1

print("when are you going to die? ", end='')
death = int(input())

for age in range(1,death):
    if age == 1:
        print("you were just born, you are",age)
    elif age > 1 and age < 40:
        print("you are young, you are",age)
    else:
        print("you are old, you are",age) 
    age = age + 1 
print("you are dead, you lived to be",age,"years old")
