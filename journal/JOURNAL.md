## 9/25/23

I spent the day working on generating random functions to train on. I already had the code to generate random functions, but now I want to run each function to see whether it halts or not programmatically rather than having to manually review every single one. The function will run in a separate thread and if it ends before 10 seconds, it'll be placed in the dataset with the "halts" flag. If it runs longer than 10 seconds, the thread will be killed and the function will be set aside in a separate file for manual review. There have been issues trying to convert the random function from a string to an actual callable function.

The [Python threading library](https://docs.python.org/3/library/threading.html) is being used to run the function in a separate thread.

<center>

![Function Flow](function_flow.png)

</center>

Here is a diagram illustrating how data will flow through the function generator when it is finished.

## 10/3/23

I realized that I was using the exec function entirely wrong. After discovering this, I made the loop to check if the function timed out or not and write it to its corresponding file. I'm also using regex to figure out the name of the function from the string. This is necessary in order to call the function as I cannot assign it to a variable. It results in somewhat ugly code, but I don't see any other solution. The functions are also being timed as they ran. I believe I could find some use for this data in the future.

## 10/5/23

I'm trying to add the break keyword to the functions but that is proving much more difficult than I thought. I have to make sure that the break is actually contained within a loop, but the structure of my program is making this difficult. The functions that generate random statements are all designed to work independently from each other but I need to know the context of the break statement in order to determine whether it is valid or not. I'm going to have to rethink the structure of my program.

## 10/11/23
The function generator can now put break statements in loops. I had to slightly change the structure of the program to make it work. The program now has a global "code" variable that can be accessed by all functions where the code is stored. The function that generates the break statements now has access to the code and can determine whether the break statement is valid or not. 

## 10/12/23
I made the loops and if statements use variables in their conditions. These variables need to have already been assigned, otherwise the program will crash. The way I found to do this is by using regex to search the global code variable for all variable assignments with the expression "var_\d*". This gets every time a variable is used, so I had to cast it to a set to make every item unique. I used this same technique to extend the assignment function to include "+=", "-=", "*=", and "/=".

## 10/13/23
I edited the conditional and loop functions to have more than one statement inside of its block. This was much harder than I anticipated, as it involved using recursion. I have to limit it to using a max of 3 statements within a block, otherwise the program will hit the maximum recursion depth.

## 10/16/23
I had to revert back to a previous version. It seems that by making the functions closer to what you would see in a regular Python function, the chance of getting a random function that never halts becomes almost zero. This is likely because most programming languages aren't built to purposefully create non-halting functions. The only way to avoid this is to limit the realism of the functions, so I reverted back before the loops and ifs used variables.

## 10/17/23
The final step in my project is just to review the potentially non-halting programs. I want to have at least 200 functions in total with a roughly even distribution for maximum training efficiency. I currently have 80 halting functions and 20 non-halting.

## 10/30/23
I just reviewed functions for an hour. I am up to 100 halting and 75 non-halting. I'm going to try to finish the review process tomorrow.

## 10/31/23
I reviewed enough functions to have 200 total with an equal split. After training the model on these, I get between 50% and 60% accuracy. I learned about a method of determining if a neural network is overfitting, so I'm going to try to use that to improve the accuracy. Essentially, I will try different parameters and log the train and test accuracies. The point where the train accuracy begins to decrease while the train accuracy increases is the point of overfitting. I will then use the parameters from before that point.

## 11/2/23
I implemented the optimization. After the test score is found, the parameters and accuracy are written to a csv for later analysis. Then, the train score is found and is also written to a csv. I think I have some issues in the optimization, as I keep getting a train score of 1.0. I'm going to try to fix this tomorrow.