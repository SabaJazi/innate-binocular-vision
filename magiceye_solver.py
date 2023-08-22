from magiceye_solve import magiceye_solver
import matplotlib.pyplot as plt 

image = plt.imread("C:\vscode\innate-binocular-vision\innate-binocular-vision\data\data.png") #load magiceye image

solution = magiceye_solver(image) #solve it
print(solution.shape)
plt.imshow(solution, cmap = plt.cm.gray) #plot the solution

plt.show() #show the plot

