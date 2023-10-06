## 9/25/23

I spent the day working on generating random functions to train on. I already had the code to generate random functions, but now I want to run each function to see whether it halts or not programmatically rather than having to manually review every single one. The function will run in a separate thread and if it ends before 10 seconds, it'll be placed in the dataset with the "halts" flag. If it runs longer than 10 seconds, the thread will be killed and the function will be set aside in a separate file for manual review. There have been issues trying to convert the random function from a string to an actual callable function.

The [Python threading library](https://docs.python.org/3/library/threading.html) is being used to run the function in a separate thread.

![Function Flow](function_flow.png)
Here is a diagram illustrating how data will flow through the function generator when it is finished.

## 10/3/23

I realized that I was using the exec function entirely wrong. After discovering this, I made the loop to check if the function timed out or not and write it to its corresponding file. I'm also using regex to figure out the name of the function from the string. This is necessary in order to call the function as I cannot assign it to a variable. It results in somewhat ugly code, but I don't see any other solution. The functions are also being timed as they ran. I believe I could find some use for this data in the future.

## 10/5/23

I'm trying to add the break keyword to the functions but that is proving much more difficult than I thought. I have to make sure that the break is actually contained within a loop, but the structure of my program is making this difficult. The functions that generate random statements are all designed to work independently from each other but I need to know the context of the break statement in order to determine whether it is valid or not. I'm going to have to rethink the structure of my program.