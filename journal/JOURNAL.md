## 9/25/23

I spent the day working on generating random functions to train on. I already had the code to generate random functions, but now I want to run each function to see whether it halts or not programmatically rather than having to manually review every single one. I'm going to run the function in a separate thread and if it ends before 10 seconds, it'll be placed in the dataset with the "halts" flag. If it runs longer than 10 seconds, the thread will be killed and the function will be set aside in a separate file for me to manually review later. I'm having trouble trying to convert the random function from a string to an actual callable function.

I'm using the [threading](https://docs.python.org/3/library/threading.html) library to run the function in a separate thread. Below is a diagram of how the function generator will work.
![Function Flow](function_flow.png)
