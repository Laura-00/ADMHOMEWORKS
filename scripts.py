#INSERTION SORT - PART 2
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the insertionSort2 function below.
def insertionSort2(n, arr):
    temp=[]
    for q in range(1,n):
        for i in range(q):
            if(arr[q] < arr[i]):
                temp=arr[q]
                arr[q]=arr[i]
                arr[i]=temp
        print(' '.join(str(x) for x in arr))
if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)
	

#INSERTION SORT - PART 2
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the insertionSort1 function below.
def insertionSort1(n, arr):
    probe = arr[-1]
    
    for ind in range(len(arr)-2, -1, -1):
        if arr[ind] > probe:
            arr[ind+1] = arr[ind]
            print(" ".join(map(str, arr)))
        else:
            arr[ind+1] = probe
            print(" ".join(map(str, arr)))
            break
    if arr[0] > probe:
        arr[0] = probe
        print(" ".join(map(str, arr)))
    

if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

#RECURSIVE DIGIT SUM
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the superDigit function below.
def superDigit(n, k):
    n=int (n)
    k=int (k)
    p = (n * k) % 9
    if p:
        return p
    else:
        return 9



if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = nk[0]

    k = int(nk[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()


#VIRAL ADVERTISING
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the viralAdvertising function below.
def viralAdvertising(n):
    people = [2]
    for i in range(n-1):
        people.append(people[-1]*3//2)
    tot=sum(people)
    return tot
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()


#NUMBER LINE JUMPS
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the kangaroo function below.
def kangaroo(x1, v1, x2, v2):
    if (v1 > v2) and not (x2 - x1) % (v2 - v1):
        return ('YES')
    else:
        return ('NO')
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()


#BIRTHDAY CAKE CANDLES 
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    count=0
    maxx=max(candles)
    for i in (candles):
        if i == maxx:
            count= count +1
    return count
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()


#POLYNOMIALS
import numpy
print(numpy.polyval(numpy.array(input().split(),float),int(input())))


#INNER AND OUTER
import numpy
a=numpy.array([input().split()], int)
b=numpy.array([input().split()], int)
print (int(numpy.inner(a,b)))
print (numpy.outer(a,b))


#DOT AND CROSS
import numpy
n=int(input())
a=numpy.array([input().split() for _ in range (n)],int)
b=numpy.array([input().split() for _ in range (n)], int)
print(numpy.dot(a,b))


#MEAN, VAR AND STD
import numpy
n,m=map(int,input().split())
arr=numpy.array([input().split() for _ in range (n)], int)
numpy.set_printoptions(sign=' ')
numpy.set_printoptions(legacy='1.13')
print (numpy.mean(arr, axis=1))
print (numpy.var(arr, axis=0))
print (numpy.std(arr, axis= None))


#MIN AND MAX
import numpy
n,m=map(int,input().split())
arr=numpy.array([input().split() for _ in range (n)], int)
minimum=numpy.min(arr, axis=1)
print(numpy.max(minimum))


#SUM AND PROD
import numpy
n,m=map(int,input().split())
arr=numpy.array([input().split() for _ in range (n)], int)
somma=numpy.sum(arr, axis=0)
print (numpy.prod(somma))


#FLOOR, CEIL AND RINT
import numpy
a=numpy.array(input().split(), float)
numpy.set_printoptions(sign=' ')
print(numpy.floor(a))
print(numpy.ceil(a))
print(numpy.rint(a))


#ARRAY MATHEMATICS
import numpy
n,m=map(int,input().split())
a=numpy.array([input().split() for i in range(n)],dtype=int)
b=numpy.array([input().split() for i in range(n)],dtype=int)
print (a+b)
print (a-b)
print (a*b)
print (a//b)
print (a%b)
print (a**b)


#EYE AND IDENTITY
import numpy
n,m=map(int,input().split())
print (str(numpy.eye(n,m,k=0)).replace('1',' 1').replace('0',' 0'))


#ZEROS AND ONES
import numpy
num = tuple(map(int, input().split()))
print (numpy.zeros(num, dtype = numpy.int))
print (numpy.ones(num, dtype = numpy.int))

#CONCATENATE
import numpy
n,m,p=map(int,input().split())
arr1 = numpy.array([input().split() for _ in range(n)],int)
arr2 = numpy.array([input().split() for _ in range(m)],int)
print(numpy.concatenate((arr1, arr2), axis = 0))


#TRANSPOSE AND FLATTEN
import numpy
n, m = map(int, input().split())
mat = numpy.array([input().strip().split() for _ in range(n)], int)
trans=numpy.transpose(mat)
print (trans)
print (mat.flatten())


#SHAPE AND RESHAPE
import numpy
print (numpy.reshape(numpy.array(input().split(),int),(3,3)))


#ARRAYS
def arrays(arr):
   return numpy.array(arr[::-1], float)
   
 
#DECORATORS 2 - NAME DIRECTORY
def person_lister(f):
    def inner(people):
        return map(f, sorted(people, key=lambda x: int(x[2])))
    return inner


#STANDARDIZE MOBILE NUMBER USING DECORATORS
def wrapper(f):
    def fun(l):
        f(["+91 "+c[-10:-5]+" "+c[-5:] for c in l])
    return fun


#XML2 - FIND THE MAXIMUM DEPTH
maxdepth = 0
def depth(elem, level):
    global maxdepth
    level += 1
    if (maxdepth < level):
        maxdepth = level
    for i in elem:
        depth(i, level)
		
		
#XML1 - FIND THE SCORE
k=[]
def get_attr_number(node):
    for elem in node.iter():
        k.append(len(elem.attrib))
    return sum(k)   


#MATRIX SCRIPT
#!/bin/python3

import math
import os
import random
import re
import sys

x,y = list(map(int,input().split()))
rows =[input() for i in range(x)]
text = "".join([row[i] for i in range(y) for row in rows])
text = re.sub('([A-Za-z1-9])[^A-Za-z1-9]+([A-Za-z1-9])', r'\1 \2', text)
text = re.sub('  ', ' ', text)
print(text)


first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)
    

#VALIDATING POSTAL CODES
regex_integer_in_range = r"^[1-9][\d]{5}$"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(\d)(?=\d\1)"	# Do not delete 'r'.


#VALIDATING UID
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re

for _ in range(int(input())):
    u = ''.join(sorted(input()))
    try:
        assert re.search(r'[A-Z]{2}', u)
        assert re.search(r'\d\d\d', u)
        assert not re.search(r'[^a-zA-Z0-9]', u)
        assert not re.search(r'(.)\1', u)
        assert len(u) == 10
    except:
        print('Invalid')
    else:
        print('Valid')


#DETECT HTML TAGS, ATTRIBUTES AND ATTRIBUTE VALUES
# Enter your code here. Read input from STDIN. Print output to STDOUT
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for attr in attrs:
            print('-> {} > {}'.format(*attr))
        
html = '\n'.join([input() for _ in range(int(input()))])
parser = MyHTMLParser()
parser.feed(html)
parser.close()


#HTML PARSER - PART 2
from html.parser import HTMLParser
import re
class MyHTMLParser(HTMLParser):
    def handle_comment(self, comment):
        if '\n' in comment:
            print('>>> Multi-line Comment')
        else:
            print('>>> Single-line Comment')
            
        print(comment)
    
    def handle_data(self, data):
        if data == '\n': return
        print('>>> Data')
        print(data)
  
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()


#HTML PARSER - PART 1
# Enter your code here. Read input from STDIN. Print output to STDOUT
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print ('Start :',tag)
        for attr in attrs:
                print ('->',' > '.join(map(str,attr)))
    def handle_endtag(self, tag):
        print ('End   :',tag)
    def handle_startendtag(self, tag, attrs):
        print ('Empty :',tag)
        for attr in attrs:
                print ('->',' > '.join(map(str,attr)))

html = ""
for i in range(int(input())):
    html += input()
                    
                
parser = MyHTMLParser()
parser.feed(html)
parser.close()



#HEX COLOR CODE
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
reg = re.compile(r"(:|,| +)(#[abcdefABCDEF1234567890]{3}|#[abcdefABCDEF1234567890]{6})\b")
n = int(input())
for i in range(n):
    line  = input()
    items = reg.findall(line)

    if items:
        for item in items:    
            print( item[1] )


#VALIDATING AND PARSING EMAIL ADDRESSES
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
import email.utils
pattern = re.compile(r"^[a-zA-Z][\w\-.]*@[a-zA-Z]+\.[a-zA-Z]{1,3}$")
for _ in range(int(input())):
    u_name, u_email = email.utils.parseaddr(input())
    if pattern.match(u_email):
        print(email.utils.formataddr((u_name, u_email)))
		

#VALIDATING PHONE NUMBERS
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
n= int(input())
for i in range(n):
    if (re.search(r"^[789]\d{9}$",input())):
        print ('YES') 
    else:
        print ('NO')


#VALIDATING ROMAN NUMERALS
regex_pattern = r"^(?=[MDCLXVI])M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"	# Do not delete 'r'.


#REGEX SUBSTITUTION
## Enter your code here. Read input from STDIN. Print output to STDOUT
import re
N = int(input())
for i in range(N):
    print (re.sub(r'(?<= )(&&|\|\|)(?= )', lambda x: 'and' if x.group() == '&&' else 'or', input()))
	

#RE.START() & RE.END()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
S = input()
k = input()
pattern = re.compile(k)
r = pattern.search(S)
if not r: print ("(-1, -1)")
while r:
    print ("({0}, {1})".format(r.start(), r.end() - 1))
    r = pattern.search(S,r.start() + 1)


#RE.FINDALL() & RE.FINDITER()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
consonanti = 'qwrtypsdfghjklzxcvbnm'
vocali = 'aeiou'
match = re.findall(r'(?<=['+consonanti+'])(['+vocali+']{2,})(?=['+consonanti+'])',input(),flags = re.I)
if match:
    for i in match:
        print (i)
else:
    print (-1)


#GROUP(), GROUPS() & GROUPDICT()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
m = re.search(r'([A-Za-z0-9])\1+',input())
if m:
    print (m.group(1))
else:
    print (-1)
	
	
#RE.SPLIT()
regex_pattern = r"[,.]"	# Do not delete 'r'.


#DETECT FLOATING POINT NUMBER
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
for i in range(int(input())):
    k = input()
    print (bool(re.match(r'^[+-]?\d*?\.{1}\d+$',k)))


#GINORTS
# Enter your code here. Read input from STDIN. Print output to STDOUT
low=[]
upper=[]
odd=[]
even=[]
for i in sorted(input()):
    if i.isalpha():
        x = upper if i.isupper() else low
    else:
        x = odd if int(i)%2 else even
    x.append(i)
print("".join(low+upper+odd+even))


#ATHLETE SORT
#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())


    s = sorted(arr, key = lambda x: x[k])

    for i in range(n):
        print (str.join(' ', map(str, s[i]))) 


#ZIPPED!
# Enter your code here. Read input from STDIN. Print output to STDOUT
n,x= list(map(int,input().split()))

C=[]
for i in range (x):
    C.append(list(map(float, input().split())))
for j in list(zip(*C)):
    print (sum(j)/len(j))


#PILING UP!
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import defaultdict,deque
T = int(input())
D = defaultdict(list)

def ordAsc(A):
    return all(A[i] <= A[i+1] for i in range(len(A)-1))

for i in range(T):
    k = input()
    D[i].append(list(map(int,input().split())))

for y in D:
    l = D[y][0].index(min(D[y][0]))
    if ordAsc(D[y][0][l:]):
        print ('Yes')
    else:
        print ('No')


#COLLECTIONS.DEQUE()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque

d = deque()
for i in range(int(input())):
    com = input().split()
    if com[0]=='append':
        d.append(int(com[1]))
    elif com[0]=='appendleft':
        d.appendleft(int(com[1]))
    elif com[0]=='pop':
        d.pop()
    else:
        d.popleft()
for i in (d):
    print (i,end=' ')


#WORD ORDER
# Enter your code here. Read input from STDIN. Print output to STDOUT
import collections;

N = int(input())
d = collections.OrderedDict()

for i in range(N):
    word = input()
    if word in d:
        d[word] +=1
    else:
        d[word] = 1

print(len(d))

for k,v in d.items():
    print(v,end = " ")


 #COLLECTIONS.ORDEREDDICT()
 # Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import OrderedDict
N = int(input());
d = OrderedDict();
for i in range(N):
    item = input().split()
    itemPrice = int(item[-1])
    itemName = " ".join(item[:-1])
    if(d.get(itemName)):
        d[itemName] += itemPrice
    else:
        d[itemName] = itemPrice
for i in d.keys():
    print(i, d[i])


#COLLECTIONS.NAMEDTUPLE()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import namedtuple
n = int(input())
col = input()
a = 0
student = namedtuple("student", col)
for i in range (n):
    stud = student._make(input().split())
    a += int(stud.MARKS)
print ("{:.2f}".format(a/n))


#DEFAULTDICT TUTORIAL
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import defaultdict

n, m = map(int, input().split())
d = defaultdict(lambda: -1)

for i in range(1, n+1): 
    word = input()
    d[word] = d[word] + ' ' + str(i) if word in d else str(i)

for _ in range(m):
    print(d[input()])


#EXCEPTIONS
for test in range(int(input())):
    try:
        a,b = map(int,input().split()) 
        division_result = a // b
        print(division_result)
    except ZeroDivisionError as e:
        print("Error Code:", e)
    except ValueError as e:
        print("Error Code:", e)


#COLLECTIONS.COUNTER()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import Counter

X = int(input())
shoes = [int(val) for val in input().split()]
N = int(input())

shoe_collection = Counter(shoes)
total_money = 0

for i in range(N):
    size, money = [int(val) for val in input().split()]

    if shoe_collection.get(size):
        total_money += money
        shoe_collection[size] -= 1

print(total_money)


#CHECK STRICT SUPERSET
# Enter your code here. Read input from STDIN. Print output to STDOUT
set_A = set(input().split())
n= int(input())
cnt = 0
check = True
for _ in range(n):
    if not set_A.issuperset(set(input().split())):
        check = False
        break
print(check)


#CHECK SUBSET
# Enter your code here. Read input from STDIN. Print output to STDOUT
test_case=int(input())
for _ in range(test_case):
    line_1, line_2 = int(input()), set(input().split())
    line_3, line_4 = int(input()), set(input().split())
    print(line_2.issubset(line_4))


#THE CAPTAIN'S ROOM
# Enter your code here. Read input from STDIN. Print output to STDOUT
k = input()
set_a= set()
set_b= set()
for room in (input().split()):
    if room not in set_a:
        set_a.add(room)
    else:
        set_b.add(room)
set_a.difference_update(set_b)
print(set_a.pop())


#SET MUTATIONS
# Enter your code here. Read input from STDIN. Print output to STDOUT
n=int(input())
a=set(map(int,input().split()))
m=int(input())
for _ in range (m):
    com=(input().split())
    s=set(map(int,input().split()))
    if com[0]=='intersection_update':
        a.intersection_update(s)
    elif com[0]=='update':
        a.update(s)
    elif com[0]=='symmetric_difference_update':
        a.symmetric_difference_update(s)
    elif com[0]=='difference_update':
        a.difference_update(s)
print (sum(map(int,a)))


#SET .SYMMETRIC_DIFFERENCE() OPERATION
# Enter your code here. Read input from STDIN. Print output to STDOUT
n =int(input())
e=set(input().split())
m=int(input())
f=set(input().split())
tot=len(e.symmetric_difference(f))
print (tot)


#SET .DIFFERENCE() OPERATION
# Enter your code here. Read input from STDIN. Print output to STDOUT
n =int(input())
e=set(input().split())
m=int(input())
f=set(input().split())
tot=len(e.difference(f))
print (tot)


#SET .INTERSECTION() OPERATION
# Enter your code here. Read input from STDIN. Print output to STDOUT
n =int(input())
e=set(input().split())
m=int(input())
f=set(input().split())
tot=len(e.intersection(f))

print (tot)


#SET .UNION() OPERATION
# Enter your code here. Read input from STDIN. Print output to STDOUT
n =int(input())
e=set(input().split())
m=int(input())
f=set(input().split())
tot=len(e.union(f))
print (tot)


#SET .DISCARD(), .REMOVE() & .POP()
n = int(input())
s = set(map(int, input().split()))
N = int(input())
for _ in range(N):
    x = input().split()
    if len(x) == 1:
        s.pop()
    elif x[0] == 'remove':
        try:
            s.remove(int(x[1]))
        except:
            next
    elif x[0] == 'discard':
        s.discard(int(x[1]))
print(sum(s))


#MAP AND LAMBDA FUNCTION
cube = lambda x: x*x*x# complete the lambda function 

def fibonacci(n):
    a = []
    for i in range(n):
        if i < 2:
            a += [i]          
        else:
            a += [a[-1] + a[-2]]
    return a


#SET .ADD()
# Enter your code here. Read input from STDIN. Print output to STDOUT
n=int(raw_input())
stamps=set(raw_input() for i in range (n))
tot=len(stamps)
print tot


#NO IDEA!
# Enter your code here. Read input from STDIN. Print output to STDOUT
n_m = map(int,raw_input().split())
n = map(int, raw_input().split())
A = set(map(int, raw_input().split()))
B = set(map(int, raw_input().split()))
happiness = 0
for i in n:
    if i in A:
        happiness += 1
    elif i in B:
        happiness -= 1
print happiness


#SYMMETRIC DIFFERENCE
# Enter your code here. Read input from STDIN. Print output to STDOUT
M = int(raw_input())
set_M = set(map(int,raw_input().split()))
N = int(raw_input())
set_N = set(map(int,raw_input().split()))
diff = (set_M.difference(set_N)).union(set_N.difference(set_M))
for i in sorted(list(diff)):
    print i


#INTRODUCTION TO SETS
def average(array):
    # your code goes here
    heights_dist = set(array)
    sum_heights_dist = sum(heights_dist)
    len_heights_dist = len(heights_dist)
    avg = sum_heights_dist/len_heights_dist
    return avg


#CALENDAR MODULE
# Enter your code here. Read input from STDIN. Print output to STDOUT
import calendar
import calendar
k = map(int,raw_input().split())
m = calendar.weekday(k[2],k[0],k[1])
w = ['MONDAY','TUESDAY','WEDNESDAY','THURSDAY','FRIDAY','SATURDAY','SUNDAY']

print w[m]


#MERGE THE TOOLS!
def merge_the_tools(string, k):
    # your code goes here
    for i in range(0,len(string), k):
        line = string[i:i+k]
        new = ""
        for i in line:
            if i not in new:
                new+=i
        print(new)



#THE MINION GAME
def minion_game(string):
    # your code goes here
    vowels_list = set(['a','e','i','o','u','A','E','I','O','U'])
    consonants = 0
    vowels = 0
 
    n = len(string)
    for i, l in enumerate(string):
        if l in vowels_list:
            vowels += n-i
        else:
            consonants += n-i
 
    if vowels == consonants:
        print "Draw"
    elif vowels > consonants:
        print "Kevin {}".format(vowels)
    else:
        print "Stuart {}".format(consonants)


#CAPITALIZE!
# Complete the solve function below.
def solve(s):
    for i in s.split():
        s = s.replace(i,i.capitalize())
    return s


#ALPHABET RANGOLI
def print_rangoli(size):
    # your code goes here
    width  = size*4-3
    string = ''

    for i in range(1,size+1):
        for j in range(0,i):
            string += chr(96+size-j)
            if len(string) < width :
                string += '-'
        for k in range(i-1,0,-1):    
            string += chr(97+size-k)
            if len(string) < width :
                string += '-'
        print(string.center(width,'-'))
        string = ''

    for i in range(size-1,0,-1):
        string = ''
        for j in range(0,i):
            string += chr(96+size-j)
            if len(string) < width :
                string += '-'
        for k in range(i-1,0,-1):
            string += chr(97+size-k)
            if len(string) < width :
                string += '-'
        print(string.center(width,'-'))


#STRING FORMATTING
def print_formatted(number):
    # your code goes here
    width = len("{0:b}".format(number)) + 1

    for i in xrange(1, number + 1):
        print "{0:d}".format(i).rjust(width - 1) + "{0:o}".format(i).rjust(width) + "{0:X}".format(i).rjust(width) + "{0:b}".format(i).rjust(width)


#DESIGNER DOOR MAT
# Enter your code here. Read input from STDIN. Print output to STDOUT
N, M = map(int,raw_input().split()) 
for i in xrange(0,N/2): 
    print ('.|.'*i).rjust((M-2)/2,'-')+'.|.'+('.|.'*i).ljust((M-2)/2,'-')
print 'WELCOME'.center(M,'-')
for i in reversed(xrange(0,N/2)): 
    print ('.|.'*i).rjust((M-2)/2,'-')+'.|.'+('.|.'*i).ljust((M-2)/2,'-')


#TEXT WRAP
def wrap(string, max_width):
    return textwrap.fill(string, max_width)


#TEXT ALIGNMENT
# Enter your code here. Read input from STDIN. Print output to STDOUT
thickness = int(input())  # This must be an odd number
c = 'H'

# Top Cone
for i in range(thickness):
    print((c * i).rjust(thickness - 1) + c + (c * i).ljust(thickness - 1))

# Top Pillars
for i in range(thickness + 1):
    print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))

# Middle Belt
for i in range((thickness + 1) // 2):
    print((c * thickness * 5).center(thickness * 6))    

# Bottom Pillars
for i in range(thickness + 1):
    print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))    

# Bottom Cone
for i in range(thickness):
    print(((c * (thickness - i - 1)).rjust(thickness) + c + (c * (thickness - i - 1)).ljust(thickness)).rjust(thickness * 6))


#STRING VALIDATORS
if __name__ == '__main__':
    s = raw_input()
    print (any(c.isalnum() for c in s))
    print (any(c.isalpha() for c in s))
    print (any(c.isdigit() for c in s))
    print (any(c.islower() for c in s))
    print (any(c.isupper() for c in s))
	

#FIND A STRING
def count_substring(string, sub_string):
    count = 0
    for i in range(len(string) - len(sub_string) + 1):
        if string[i:i+len(sub_string)] == sub_string:
            count += 1
    return count


#MUTATIONS
def mutate_string(string, position, character):
    l=list(string)
    l[position]=character
    string="".join(l)
    return string


#WHAT'S YOUR NAME?
def print_full_name(a, b):
    print ('Hello '+a+' '+b+'! You just delved into python.')
   

#STRING SPLIT AND JOIN
def split_and_join(line):
    line=line.split(" ")
    line="-".join(line)
    return line


#SWAP CASE
def swap_case(s):
    return s.swapcase()


#TUPLES
if __name__ == '__main__':
    n = int(raw_input())
    integer_list = map(int, raw_input().split())
    t = tuple(integer_list)
    print(hash(t))


#LISTS
if __name__ == '__main__':
    N = int(raw_input())
    
    lista = []
    
    for i in range(N): 
        option = raw_input().split()
        if option[0] == 'print':
            print(lista)
        elif option[0] == 'sort':
            lista.sort()
        elif option[0] == 'remove':
            lista.remove(int(option[1]))
        elif option[0] == 'append':
            lista.append(int(option[1]))
        elif option[0] == 'insert':
            lista.insert(int(option[1]),int(option[2]))
        elif option[0] == 'reverse':
            lista.reverse()
        elif option[0] == 'pop':
            lista.pop()


#VALIDATING CREDIT CARD NUMBERS
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
for i in range(int(raw_input())):
    S = raw_input().strip()
    pre_match = re.search(r'^[456]\d{3}(-?)\d{4}\1\d{4}\1\d{4}$',S)
    if pre_match:
        processed_string = "".join(pre_match.group(0).split('-'))
        final_match = re.search(r'(\d)\1{3,}',processed_string)
        print 'Invalid' if final_match else 'Valid'
    else:
        print 'Invalid'


#LINEAR ALGEBRA
# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy
n=int(raw_input())
m=[]
for i in range(n):
    m.append(map(float,raw_input().split()))
print round(numpy.linalg.det(numpy.array(m)),2)


#FINDING THE PERCENTAGE
if __name__ == '__main__':
    n = int(raw_input())
    student_marks = {}
    for _ in range(n):
        line = raw_input().split()
        name, scores = line[0], line[1:]
        scores = map(float, scores)
        student_marks[name] = scores
    query_name = raw_input()
    l=list(student_marks[query_name])
    length=len(l)
    s=sum(l)
    average=s/length
    print('%.2f'%average)


#LIST COMPREHENSIONS
if __name__ == '__main__':
    x = int(raw_input())
    y = int(raw_input())
    z = int(raw_input())
    n = int(raw_input())
    result=[[i,j,k] for i in range (x+1) for j in range (y+1) for k in range (z+1) if (i+j+k!=n)]

    print (result)


#NESTED LISTS
from __future__ import print_function
score_list = {}
for _ in range(input()):
    name = raw_input()
    score = float(raw_input())
    if score in score_list:
        score_list[score].append(name)
    else:
        score_list[score] = [name]
new_list = []
for i in score_list:
    new_list.append([i, score_list[i]])
new_list.sort()
result = new_list[1][1]
result.sort()
print (*result, sep = "\n")


#FIND THE RUNNER-UP SCORE!
if __name__ == '__main__':
    n = int(raw_input())
    arr = map(int, raw_input().split())
    
    s1 = max(arr)
    s2 = -9999999999
    for i in range(n):
        if arr[i] != s1 and arr[i] > s2:
            s2 = arr[i]
    print s2


#COMPRESS THE STRING!
# Enter your code here. Read input from STDIN. Print output to STDOUT
from __future__ import print_function
from itertools import *

for i,j in groupby(map(int,list(raw_input()))):
    print(tuple([len(list(j)), i]) ,end = " ")


#COMPANY LOGO
# Enter your code here. Read input from STDIN. Print output to STDOUT
S = raw_input()
letters = [0]*26

for letter in S:
    letters[ord(letter)-ord('a')] += 1

for _ in range(3):
    
    max_letter = max(letters)
    
    for index in range(26):
        if max_letter == letters[index]:
            print chr(ord('a')+index), max_letter
            letters[index] = -1
            break


#PRINT FUNCTION
from __future__ import print_function

if __name__ == '__main__':
    n = int(input())
    for i in range (n):
        print (i+1,end="")


#WRITE A FUNCTION
def is_leap(year):
    leap = False
    
    # Write your logic here
    if (year%4==0):
        leap= True
        if (year%100==0):
            leap = False
            if (year%400==0):
                leap = True
    return leap


#LOOPS
if __name__ == '__main__':
    n = int(raw_input())
    for i in range (n) :
        if i<n:
            print (i*i)


#PYTHON: DIVISION
from __future__ import division

if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())
print (a//b)
print (a/b)


#ARITHMETIC OPERATORS
if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())
print (a+b)
print (a-b)
print (a*b)


#PYTHON IF-ELSE
#!/bin/python

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(raw_input().strip())
    
    if n in range (2,5):
        if n%2==0:
            print ("Not Weird")
        else: print ("Weird")
    elif n in range (6,20):
        #if n%2==0:
            #print ("Not Weird")
        #else: print ("Weird")
        print ("Weird")
    elif n > 20:
        if n%2==0:
            print ("Not Weird")
        else: print ("Weird")
    else : print ("Weird")


#SAY "HELLO, WORLD!" WITH PYTHON
if __name__ == '__main__':
    print "Hello, World!"

#SOLVE ME FIRST
def solveMeFirst(a,b):
	# Hint: Type return a+b below
    return a+b

num1 = int(input())
num2 = int(input())
res = solveMeFirst(num1,num2)
print(res)
