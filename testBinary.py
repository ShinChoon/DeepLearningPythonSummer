import numpy as np


def bin_float(reciv_str):
    #remove decimal
    digit_location = reciv_str.find('.')
    if digit_location != -1:
        clip_str = reciv_str[(digit_location+1):]
    else:
        clip_str = reciv_str
    P_flag = False if clip_str[0] == '1' else True
    str_num = clip_str[1:]
    reverse_num = str_num[::-1]
    answer = 0.0
    factor = 0
    for i in reverse_num:

        answer = answer + int(i) * 0.0625 * 2**factor
        factor = factor + 1

    factor = 0
    if digit_location != -1:
        for i in reciv_str[0:digit_location]:
            answer = answer + int(i) * 1 * 2**factor
            factor = factor + 1

    if not P_flag:
        answer = -1 * answer

    return answer



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

    for x in range(int(places-1)):
        
        answer = (decimal_converter(float(dec))) * 2
        # Convert the decimal part
        # to float 4-digit again
        whole, _dec = str(answer).split(".")
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