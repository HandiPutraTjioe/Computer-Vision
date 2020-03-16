# import re 

# str = "The rain in Spain"
# x = re.search("^The.*Spain$", str)

# if (x):
#     print("YES! We have a match!") 
# else:
#     print("No match")

# print(x.span())



# import re 

# str = "8 times before 11:45 AM"

# x = re.findall("[a-z]", str)

# print(x)



# import re 

# str = "My name is Handi"

# x = re.split("[\s]", str)

# print(x)


import re 

str = "My name is Handi"

x = re.sub("\s", "9", str)

print(x)