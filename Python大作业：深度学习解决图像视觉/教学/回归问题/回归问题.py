#对y = wx + b进行拟合
#求误差

import numpy as np

def computer_error_for_line_given_points(b,w,points):
    #points是一系列x,y值的集合，也就是一个二维数组
    #也就是计算loss，或者说方差
    totalerror = 0
    for i in range(0 , len(points)):
        x = points[i , 0]
        y = points[i , 1]
        totalerror += (y - (w * x + b)) ** 2
    return totalerror / float(len(points))

#求梯度，并进行拟合修正
#即使求出的w,b函数方差尽可能小
def step_g(b_c,w_c,points,learningrate):
    b_g = 0
    w_g = 0
    n = float(len(points))#求points的大小

    for i in range(0 , len(points)):
        x = points[i , 0]
        y = points[i , 1]
        b_g += (2 / n) * (w_c * x + b_c - y)#求b的平均梯度
        w_g += (2 / n) * (w_c * x + b_c - y) * x#求w的平均梯度
    
    new_b = b_c - (learningrate * b_g) #返回下一个b
    new_w = w_c - (learningrate * w_g) #返回下一个w
    return [new_b,new_w]

#进行循环迭代
def gradient_descent_runner(points,start_b,start_w,
                            learningrate,num_iterations):
    b = start_b
    w = start_w

    for i in range(num_iterations):
        b,w = step_g(b, w, np.array(points), learningrate)
    return [b , w]

def run():
    points = np.genfromtxt("/Users/lixiaoyi/Desktop/大作业/Python大作业：深度学习解决图像视觉/教学/回归问题/data3.csv" , delimiter = ",")
    learning_rate = 0.0001
    initial_b = 0
    initial_w = 0
    #最开始猜测的b,w的值
    num_iterations = 50000
    print("starting gradient descent at b = {0}, w = {1}, error = {2}",
            initial_b, initial_w,
                    computer_error_for_line_given_points(initial_b, initial_w, points))
    
    print("running...")

    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)

    print("after {0} iterations b = {1}, w = {2}, error = {3}",
            num_iterations, b ,w,
                    computer_error_for_line_given_points(b , w ,points))
    print(w,b)
        
if __name__ == '__main__':
    run()