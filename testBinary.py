import numpy as np

# def float_bin(number, places=4):
#     source = float(number)
#     N_flag = True if source<=0 else False
#     _number = source if source >= 0 else -1*source
#     whole, dec = str(number).split(".")

#     dec = int(dec)
#     whole = int(whole)

#     dec = _number - int(whole)

#     res = bin(0).lstrip("0b")
#     if whole > 0:
#     #detect if any value more than 1
#         res = bin(whole).lstrip("0b") + "."
#     else:
#         res = bin(0).lstrip("0b")

#     for x in range(places):
        
#         answer = (decimal_converter(float(dec))) * 2
#         # Convert the decimal part
#         # to float 4-digit again
#         answer = float("{:.4f}".format(answer))
#         whole, _dec = str(answer).split(".")

#         print("answer: ", answer)
#         print("whole: ", whole)
#         if answer > 0:
#             dec = answer - int(whole)

#         # Keep adding the integer parts
#         # receive to the result variable
#         res += whole
 
#     result = str(res)

#     if N_flag:
#         result = '1' + result
#     else:
#         result = '0' + result

#     return result



def float_bin(number, places):
    number = float(number)
    if np.isnan(number):
        number =  0
    source = float("{:.4f}".format(number))   
    N_flag = True if source<=0 else False
    _number = source if source >= 0 else -1*source
    whole, dec = str(source).split(".")

    dec = int(dec)
    whole = int(whole)

    dec = _number - int(whole)

    res = bin(0).lstrip("0b")
    if whole > 0:
    #detect if any value more than 1
        res = bin(whole).lstrip("0b") + "."
    else:
        res = bin(0).lstrip("0b")

    for x in range(int(places)):
        
        answer = (decimal_converter(float(dec))) * 2
        # Convert the decimal part
        # to float 4-digit again
        whole, _dec = str(answer).split(".")
        print("answer: ", answer)
        print("whole: ", whole)
        if answer > 0:
            dec = answer - int(whole)
        else:
            whole, _dec = str(0.0).split(".")

        # Keep adding the integer parts
        # receive to the result variable
        res += whole
 
    result = str(res)

    if N_flag:
        result = '1' + result
    else:
        result = '0' + result
  
    return result

def decimal_converter(num):
    while num > 1:
        num /= 10
    return num

n = input("Enter your floating point value : \n") 
bits = input("Enter your bits : \n")
# Take user input for the number of
# decimal places user want result as
 
print(float_bin(n,bits))