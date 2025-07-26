textbook_examples = [
    {
        "problem": "P and Q each have a certain number of apples. A says to B, “If you give me 10 of your apples, I will have twice the number of apples left with you.” B replies, “If you give me 10 of your apples, I will have the same number of apples as left with you.” Find the number of apples with P and Q separately.",
        "reasoning": """
        1. Let x be the number of apples with P and y be the number of apples with Q.
        2. If B gives A 10 apples, then A has x + 10 and B has y - 10.
        3. According to A's statement, x + 10 = 2(y - 10).
        4. If A gives B 10 apples, then A has x - 10 and B has y + 10.
        5. According to B's statement, x - 10 = y + 10.
        Final Equations: 
        x + 10 = 2(y - 10) and x - 10 = y + 10
        """
    },
    {
        "problem": "On selling a T.V. at 5% gain and a fridge at 10% gain, Reliance Digital gains Rs 2000. But if it sells the T.V. at 10% gain and the fridge at 5% loss, he gains Rs 1500 on the transaction. Find the actual prices of T.V. and fridge.",
        "reasoning": """
        1. Let x be the cost price of the T.V. and y be the cost price of the fridge.
        2. Selling the T.V. at 5% gain gives 0.05x, and selling the fridge at 10% gain gives 0.10y.
        3. The total gain from selling at these rates is 0.05x + 0.10y = 2000.
        4. In the second scenario, the gain from selling the T.V. at 10% is 0.10x, and the loss from the fridge at 5% loss is -0.05y.
        5. The total gain in this case is 0.10x - 0.05y = 1500.
        Final Equations: 
        0.05x + 0.10y = 2000 and 0.10x - 0.05y = 1500
        """
    },
    {
        "problem": "A and B are friends and their ages differ by 2 years. A's father D is twice as old as A and B is twice as old as his sister C. The age of D and C differ by 40 years. Find the ages of A and B.",
        "reasoning": """
        1. Let x be the age of A and y be the age of B.
        2. The difference in their ages gives x - y = 2.
        3. Since D is twice as old as A, we have D = 2x.
        4. If B is twice as old as his sister C, we express C's age as C = y / 2.
        5. The age difference between D and C gives 2x - (y / 2) = 40.
        Final Equations: 
        x - y = 2 and 2x - (y / 2) = 40
        """
    },
    {
        "problem": "Five years hence, father's age will be three times the age of his son. Five years ago, the father was seven times as old as his son. Find their present ages.",
        "reasoning": """
        1. Let x be the father's age and y be the son's age.
        2. In 5 years, the father's age will be x + 5 and the son's age will be y + 5.
        3. Setting up the equation from the first condition, x + 5 = 3(y + 5).
        4. Five years ago, the father's age was x - 5 and the son's age was y - 5.
        5. From this condition, we have x - 5 = 7(y - 5).
        Final Equations: 
        x + 5 = 3(y + 5) and x - 5 = 7(y - 5)
        """
    },
    {
        "problem": "The ages of two friends Manjit and Ranjit differ by 3 years. Manjit's father Dharam is twice as old as Manjit and Ranjit is twice as old as his sister Jaspreet. The ages of Jaspreet and Dharam differ by 30 years. Find the ages of Manjit and Ranjit.",
        "reasoning": """
        1. Let x be the age of Manjit and y be the age of Ranjit.
        2. The age difference gives x - y = 3.
        3. Dharam's age is 2x, and Ranjit's sister's age is C = y / 2.
        4. The difference in ages between Dharam and Jaspreet gives 2x - (y / 2) = 30.
        Final Equations: 
        x - y = 3 and 2x - (y / 2) = 30
        """
    },
    {
        "problem": "A takes 3 hours more than B to walk 30 km. But if A doubles his pace, he is ahead of B by one and a half hours. Find their speed of walking.",
        "reasoning": """
        1. Let x be the speed of A and y be the speed of B.
        2. The time taken by A to walk 30 km is 30/x and by B is 30/y.
        3. From the problem statement, (30/x) - (30/y) = 3.
        4. If A doubles his speed, he travels at 2x, leading to (30/y) - (30/(2x)) = 1.5.
        Final Equations: 
        (30/x) - (30/y) = 3 and (30/y) - (30/(2x)) = 1.5
        """
    },
    {
        "problem": "The boat goes 30 km upstream and 44 km downstream in 10 hours. In 13 hours, it can go 40 km upstream and 55 km downstream. Determine the speed of stream and that of the boat in still water.",
        "reasoning": """
        1. Let x be the speed of the boat in still water and y be the speed of the stream.
        2. The effective speed upstream is (x - y) and downstream is (x + y).
        3. From the first scenario: (30/(x - y)) + (44/(x + y)) = 10.
        4. From the second scenario: (40/(x - y)) + (55/(x + y)) = 13.
        Final Equations: 
        (30/(x - y)) + (44/(x + y)) = 10 and (40/(x - y)) + (55/(x + y)) = 13
        """
    },
    {
        "problem": "A boat goes 24 km upstream and 28 km downstream in 6 hours. It goes 30 km upstream and 21 km downstream in 6 hours and 30 minutes. Find the speed of the boat in still water and also speed of the stream.",
        "reasoning": """
        1. Let x be the speed of the boat in still water and y be the speed of the stream.
        2. The equations can be set up as: (24/(x - y)) + (28/(x + y)) = 6.
        3. From the second scenario: (30/(x - y)) + (21/(x + y)) = 6.5.
        Final Equations: 
        (24/(x - y)) + (28/(x + y)) = 6 and (30/(x - y)) + (21/(x + y)) = 6.5
        """
    },
    {
        "problem": "A man walks a certain distance with a certain speed. If he walks 1/2 km an hour faster, he takes 1 hour less. But, if he walks 1 km an hour slower, he takes 3 more hours. Find the distance covered by the man and his original rate of walking.",
        "reasoning": """
        1. Let v be the original speed, t be the time taken, and d be the distance.
        2. For the faster case: vt = (v + 0.5)(t - 1).
        3. For the slower case: vt = (v - 1)(t + 3).
        Final Equations: 
        vt = (v + 0.5)(t - 1) and vt = (v - 1)(t + 3)
        """
    },
    {
        "problem": "Anuj travels 600 km partly by train and partly by car. If he covers 400 km by train and the rest by car, it takes him 6 hours and 30 minutes. But, if he travels 200 km by train and the rest by car, he takes half an hour longer. Find the speed of the train and that of the car.",
        "reasoning": """
        1. Let x be the speed of the train and y be the speed of the car.
        2. From the first scenario: (400/x) + (200/y) = 6.5.
        3. From the second scenario: (200/x) + (400/y) = 7.
        Final Equations: 
        (400/x) + (200/y) = 6.5 and (200/x) + (400/y) = 7
        """
    },
    {
        "problem": "A man has a certain number of apples. If he gives 15 apples to his friend, he will have 5 times the number of apples his friend has. If he takes 15 apples from his friend, he will have 3 times the number of apples his friend has. Find the number of apples with the man and his friend.",
        "reasoning": """
        1. Let x be the number of apples the man has and y be the number of apples the friend has.
        2. After giving 15 apples, the equation becomes: x - 15 = 5(y + 15).
        3. After taking 15 apples, the equation becomes: x + 15 = 3(y - 15).
        Final Equations: 
        x - 15 = 5(y + 15) and x + 15 = 3(y - 15)
        """
    },
    {
        "problem": "The sum of the ages of three brothers is 45 years. If the age of the youngest brother is x, the middle brother is y, and the oldest brother is z. The oldest brother is 5 years older than the middle brother, and the middle brother is 3 years older than the youngest. Find the ages of the three brothers.",
        "reasoning": """
        1. The equation for the sum of their ages is: x + y + z = 45.
        2. From the age relationships: z = y + 5 and y = x + 3.
        Final Equations: 
        x + y + z = 45, z = y + 5, y = x + 3
        """
    },
    {
        "problem": "A sum of money is invested at 10% per annum simple interest. If it were invested at 12% per annum, it would have earned Rs. 100 more in 2 years. Find the principal amount.",
        "reasoning": """
        1. Let P be the principal amount.
        2. The interest at 10% for 2 years is: I1 = (10/100) * P * 2.
        3. The interest at 12% for 2 years is: I2 = (12/100) * P * 2.
        4. The difference between the two interests gives: I2 - I1 = 100.
        Final Equations: 
        I1 = (10/100) * P * 2, I2 = (12/100) * P * 2, I2 - I1 = 100
        """
    },
    {
        "problem": "The perimeter of a rectangle is 50 meters. If the length is 5 meters more than the breadth, find the dimensions of the rectangle.",
        "reasoning": """
        1. Let l be the length and b be the breadth.
        2. The equation for the perimeter is: 2(l + b) = 50.
        3. The relationship between length and breadth is: l = b + 5.
        Final Equations: 
        2(l + b) = 50, l = b + 5
        """
    },
    {
        "problem": "In a class, the average age of students is 18 years. If the teacher’s age is included, the average age becomes 19 years. If the number of students is n, find the teacher's age.",
        "reasoning": """
        1. The total age of students is: 18n.
        2. With the teacher included, the total age becomes: 18n + T.
        3. The average age with the teacher is: (18n + T)/(n + 1) = 19.
        Final Equations: 
        18n + T = 19(n + 1)
        """
    },
    {
        "problem": "The difference between two numbers is 20, and when one number is divided by the other, the quotient is 3. Find the numbers.",
        "reasoning": """
        1. Let x and y be the two numbers.
        2. The difference gives: x - y = 20.
        3. The quotient gives: x/y = 3.
        Final Equations: 
        x - y = 20, x/y = 3
        """
    },
    {
        "problem": "A chemist has 30 liters of a solution that is 40% acid. He wants to dilute it to create a solution that is 25% acid. How much water should he add?",
        "reasoning": """
        1. The amount of acid in the original solution is: 0.4 * 30.
        2. Let w be the liters of water to be added. The new concentration can be expressed as: 0.4 * 30 / (30 + w) = 0.25.
        Final Equations: 
        0.4 * 30 / (30 + w) = 0.25
        """
    },
    {
        "problem": "A rectangular plot has a length that is twice its width. If the perimeter of the plot is 60 meters, what are the dimensions of the plot?",
        "reasoning": """
        1. Let w be the width and l be the length.
        2. The length can be expressed as: l = 2w.
        3. The perimeter gives: 2(l + w) = 60.
        Final Equations: 
        2(l + w) = 60, l = 2w
        """
    },
    {
        "problem": "A sum of Rs. 5000 is divided into two parts such that the first part earns 8% per annum and the second part earns 6% per annum. If the total interest earned in one year is Rs. 360, find the amount in each part.",
        "reasoning": """
        1. Let x be the amount in the first part and y be the amount in the second part.
        2. The total amount gives: x + y = 5000.
        3. The interest gives: 0.08x + 0.06y = 360.
        Final Equations: 
        x + y = 5000, 0.08x + 0.06y = 360
        """
    },
    {
        "problem": "Two trains start from the same station at the same time, one travelling north and the other south. If the first train travels at 60 km/h and the second at 90 km/h, how far apart will they be after 2 hours?",
        "reasoning": """
        1. The distance traveled by the first train in 2 hours is: d1 = 60 * 2.
        2. The distance traveled by the second train in 2 hours is: d2 = 90 * 2.
        3. The total distance apart is: d1 + d2.
        Final Equations: 
        d1 = 60 * 2, d2 = 90 * 2, total distance = d1 + d2
        """
    },
    {
        "problem": "A two-digit number is such that the sum of the number and the number obtained by reversing its digits is 121. If the difference between the digits of the number is 3, find the number.",
        "reasoning": """
        1. Let the two-digit number be represented as 10a + b, where a is the tens digit and b is the units digit.
        2. The reverse of the number is 10b + a.
        3. According to the problem, we have the equation: (10a + b) + (10b + a) = 121.
        4. This simplifies to: 11a + 11b = 121.
        5. Dividing by 11 gives: a + b = 11. (Equation 1)
        6. The problem states that the difference between the digits is 3, leading to: a - b = 3. (Equation 2)
        7. From Equation 1: b = 11 - a.
        8. Substitute into Equation 2: a - (11 - a) = 3.
        Final Equations: 
        a + b = 11, a - b = 3
        """
    },
    {
        "problem":"n selling a T.V. at 5% gain and a fridge at 10% gain, Reliance Digital gains Rs 2000. But if it sells the T.V. at 10% gain and the fridge at 5% loss, he gains Rs 1500 on the transaction. Find the actual prices of T.V. and fridge.",
        "reasoning":"""
        1. Let x be the cost price of the T.V. and y be the cost price of the fridge.
        2. Selling the T.V. at 5% gain gives 0.05x, and selling the fridge at 10% gain gives 0.10y.
        3. The total gain from selling at these rates is 0.05x + 0.10y = 2000.
        4. In the second scenario, the gain from selling the T.V. at 10% is 0.10x, and the loss from the fridge at 5% loss is -0.05y.
        5. The total gain in this case is 0.10x - 0.05y = 1500.
        Final Equations: 
        0.05x + 0.10y = 2000 and 0.10x - 0.05y = 1500"""
    },
    {
        "problem": "Jane is organizing a fundraiser. The total number of adults and children attending is 80. The entrance fee for adults is $10, and for children, it is $5. The total amount collected is $600. Write a system of linear equations to represent this information and find the number of adults and children.",
        "reasoning": """
        1. Let a be the number of adults and c be the number of children attending.
        2. The total number of adults and children attending is given by the equation:
           a + c = 80
        3. The entrance fee for adults is $10, and for children, it is $5. The total amount collected is:
           10a + 5c = 600
        Final Equations: 
           a + c = 80 and 10a + 5c = 600
        """
    },
    {
        "problem": "Alice has $50 and decides to save $5 each week. Bob has no savings initially but saves $8 each week. After how many weeks will Alice and Bob have the same amount of money saved?",
        "reasoning": """
        1. Let w represent the number of weeks.
        2. Alice has $50 and saves $5 each week, giving her total savings as:
           50 + 5w
        3. Bob has no savings initially but saves $8 each week, giving his total savings as:
           8w
        4. To find out after how many weeks they will have the same amount of money saved:
           50 + 5w = 8w
        Final Equations: 
           50 + 5w = 8w
        """
    },
    {
        "problem": "The sum of two numbers is 15. Twice the first number added to three times the second number is 36. Write a system of linear equations to represent this information and find the values of the two numbers.",
        "reasoning": """
        1. Let x be the first number and y be the second number.
        2. The sum of the two numbers is:
           x + y = 15
        3. Twice the first number added to three times the second number is:
           2x + 3y = 36
        Final Equations: 
           x + y = 15 and 2x + 3y = 36
        """
    },
    {
        "problem": "The present ages of two friends are such that the sum of their ages is 50. Six years ago, one friend's age was twice the age of the other. Determine their present ages.",
        "reasoning": """
        1. Let x be the present age of the first friend and y be the present age of the second friend.
        2. The sum of their ages is:
           x + y = 50
        3. Six years ago, one friend's age was twice the age of the other:
           x - 6 = 2(y - 6)
        Final Equations: 
           x + y = 50 and x - 6 = 2(y - 6)
        """
    },
    {
        "problem": "A and B can complete a work together in 15 days. If A works alone for 5 days, and then B finishes the remaining work in 10 days. How long will A take to complete the work alone?",
        "reasoning": """
        1. Let t be the time taken by A to complete the work alone.
        
        2. The work done by A in one day is:
           Rate of A = 1/t (work/day).
           
        3. The work done by B in one day is:
           Rate of B = 1/15 - 1/t (work/day).
           
        4. In 5 days, A completes:
           Work done by A = 5 * (1/t).
           
        5. The remaining work is completed by B in 10 days:
           Work done by B = 10 * (1/t - 1/15).
           
        Final Equations:
           5*(1/t) + 10*(1/t - 1/15) = 1
        """
    },
    {
        "problem": "X can complete a work in 12 days, Y can complete it in 15 days, and Z can complete it in 10 days. If X works for 3 days, then Y takes over and works for 2 days, and finally, Z finishes the work. How long does Z take to finish the work?",
        "reasoning": """
        1. Let the total work be represented as 1 unit.
        
        2. The rate of work done by X is:
           Rate of X = 1/12 (work/day).

        3. The rate of work done by Y is:
           Rate of Y = 1/15 (work/day).

        4. The rate of work done by Z is:
           Rate of Z = 1/10 (work/day).

        5. In 3 days, X completes:
           Work done by X = 3 * (1/12).

        6. In 2 days, Y completes:
           Work done by Y = 2 * (1/15).

        7. The total work done by X and Y combined is:
           Total Work Done = Work done by X + Work done by Y.

        8. The remaining work for Z to finish is:
           Remaining Work = 1 - Total Work Done.

        9. Let t be the time taken by Z to finish the remaining work:
           Remaining Work = t * (1/10).

        Final Equations:
           1 - (3*(1/12) + 2*(1/15)) = t*(1/10).
        """
    },
    {
        "problem": "The sum of two numbers is 24. The first number is 4 more than twice the second number. Write a system of linear equations to represent this information and find the values of the two numbers.",
        "reasoning": """
        1. Let x be the first number and y be the second number.
        2. The sum of the two numbers is:
           x + y = 24
        3. The first number is 4 more than twice the second number:
           x = 2y + 4
        Final Equations: 
           x + y = 24 and x = 2y + 4
        """
    },
    {
        "problem": "A father is 4 times as old as his son. The sum of their ages is 60 years. Write a system of linear equations to represent this information and find their ages.",
        "reasoning": """
        1. Let x be the father's age and y be the son's age.
        2. The father is 4 times as old as his son:
           x = 4y
        3. The sum of their ages is 60:
           x + y = 60
        Final Equations: 
           x = 4y and x + y = 60
        """
    },
    {
        "problem": "A boat travels downstream for 4 hours and covers 32 miles. It travels upstream for 4 hours and covers 24 miles. Write a system of linear equations to represent this information and find the speed of the boat in still water and the speed of the current.",
        "reasoning": """
        1. Let x be the speed of the boat in still water and y be the speed of the current.
        2. The boat travels downstream, so its effective speed is (x + y):
           4(x + y) = 32
        3. The boat travels upstream, so its effective speed is (x - y):
           4(x - y) = 24
        Final Equations: 
           4(x + y) = 32 and 4(x - y) = 24
        """
    },
    {
        "problem": "A man can complete a task in 6 hours working alone, but if he works with a colleague, they can complete the task in 4 hours. Write a system of linear equations to represent this information and find the time it would take for his colleague to complete the task alone.",
        "reasoning": """
        1. Let x be the time it takes the man to complete the task alone, and y be the time it takes his colleague.
        2. The man's rate of work is 1/6 (since he finishes in 6 hours):
           1/x = 1/6
        3. Working together, their combined rate is 1/4 (since they finish in 4 hours):
           1/x + 1/y = 1/4
        Final Equations: 
           1/x = 1/6 and 1/x + 1/y = 1/4
        """
    },
    {
        "problem": "A train travels 120 miles downstream in 2 hours and 80 miles upstream in 2 hours. Write a system of linear equations to represent this information and find the speed of the train in still water and the speed of the current.",
        "reasoning": """
        1. Let x be the speed of the train in still water and y be the speed of the current.
        2. The effective speed downstream is (x + y):
           2(x + y) = 120
        3. The effective speed upstream is (x - y):
           2(x - y) = 80
        Final Equations: 
           2(x + y) = 120 and 2(x - y) = 80
        """
    },
    {
        "problem": "The sum of two numbers is 18. The first number is 2 more than three times the second number. Write a system of linear equations to represent this information and find the values of the two numbers.",
        "reasoning": """
        1. Let x be the first number and y be the second number.
        2. The sum of the two numbers is:
           x + y = 18
        3. The first number is 2 more than three times the second number:
           x = 3y + 2
        Final Equations: 
           x + y = 18 and x = 3y + 2
        """
    },
    {
        "problem": "A man’s age is three years less than twice his son’s age. The sum of their ages is 51. Write a system of linear equations to represent this information and find their ages.",
        "reasoning": """
        1. Let x be the man's age and y be the son's age.
        2. The man's age is three years less than twice his son's age:
           x = 2y - 3
        3. The sum of their ages is 51:
           x + y = 51
        Final Equations: 
           x = 2y - 3 and x + y = 51
        """
    },
    {
        "problem": "The sum of two numbers is 70. The first number is 5 more than twice the second number. Write a system of linear equations to represent this information and find the values of the two numbers.",
        "reasoning": """
        1. Let x be the first number and y be the second number.
        2. The sum of the two numbers is:
           x + y = 70
        3. The first number is 5 more than twice the second number:
           x = 2y + 5
        Final Equations: 
           x + y = 70 and x = 2y + 5
        """
    },
    {
        "problem": "A father is 6 years older than three times the age of his son. The sum of their ages is 72 years. Write a system of linear equations to represent this information and find their ages.",
        "reasoning": """
        1. Let x be the father's age and y be the son's age.
        2. The father is 6 years older than three times the age of his son:
           x = 3y + 6
        3. The sum of their ages is 72:
           x + y = 72
        Final Equations: 
           x = 3y + 6 and x + y = 72
        """
    },
    {
        "problem": "A car travels 180 miles downstream in 3 hours and 120 miles upstream in 3 hours. Write a system of linear equations to represent this information and find the speed of the car in still water and the speed of the current.",
        "reasoning": """
        1. Let x be the speed of the car in still water and y be the speed of the current.
        2. The effective speed downstream is (x + y):
           3(x + y) = 180
        3. The effective speed upstream is (x - y):
           3(x - y) = 120
        Final Equations: 
           3(x + y) = 180 and 3(x - y) = 120
        """
    },
    {
        "problem": "A boat covers a distance of 30 miles downstream in 2 hours and 20 miles upstream in 2 hours. Write a system of linear equations to represent this information and find the speed of the boat in still water and the speed of the current.",
        "reasoning": """
        1. Let x be the speed of the boat in still water and y be the speed of the current.
        2. The effective speed downstream is (x + y):
           2(x + y) = 30
        3. The effective speed upstream is (x - y):
           2(x - y) = 20
        Final Equations: 
           2(x + y) = 30 and 2(x - y) = 20
        """
    },
    {
        "problem": "The sum of two numbers is 25. The difference between five times the first number and three times the second number is 50. Write a system of linear equations to represent this information and find the values of the two numbers.",
        "reasoning": """
        1. Let x be the first number and y be the second number.
        2. The sum of the two numbers is:
           x + y = 25
        3. The difference between five times the first number and three times the second number is:
           5x - 3y = 50
        Final Equations: 
           x + y = 25 and 5x - 3y = 50
        """
    },
    {
        "problem": "A mother is 8 years older than three times the age of her daughter. The sum of their ages is 68 years. Write a system of linear equations to represent this information and find their ages.",
        "reasoning": """
        1. Let x be the mother's age and y be the daughter's age.
        2. The mother is 8 years older than three times the age of her daughter:
           x = 3y + 8
        3. The sum of their ages is 68:
           x + y = 68
        Final Equations: 
           x = 3y + 8 and x + y = 68
        """
    },
    {
        "problem": "A train travels 150 miles downstream in 3 hours and 100 miles upstream in 4 hours. Write a system of linear equations to represent this information and find the speed of the train in still water and the speed of the current.",
        "reasoning": """
        1. Let x be the speed of the train in still water and y be the speed of the current.
        2. The effective speed downstream is (x + y):
           3(x + y) = 150
        3. The effective speed upstream is (x - y):
           4(x - y) = 100
        Final Equations: 
           3(x + y) = 150 and 4(x - y) = 100
        """
    },
    {
        "problem": "A father’s age is 5 years more than four times his daughter’s age. The sum of their ages is 65. Write a system of linear equations to represent this information and find their ages.",
        "reasoning": """
        1. Let x be the father's age and y be the daughter's age.
        2. The father's age is 5 years more than four times the daughter's age:
           x = 4y + 5
        3. The sum of their ages is 65:
           x + y = 65
        Final Equations: 
           x = 4y + 5 and x + y = 65
        """
    },
    {
        "problem": "A woman is 3 times as old as her son. The sum of their ages is 48. Write a system of linear equations to represent this information and find their ages.",
        "reasoning": """
        1. Let x be the woman's age and y be the son's age.
        2. The woman is 3 times as old as her son:
           x = 3y
        3. The sum of their ages is 48:
           x + y = 48
        Final Equations: 
           x = 3y and x + y = 48
        """
    },
    {
        "problem": "The sum of two numbers is 12. The first number is 3 more than twice the second number. Write a system of linear equations to represent this information and find the values of the two numbers.",
        "reasoning": """
        1. Let x be the first number and y be the second number.
        2. The sum of the two numbers is:
           x + y = 12
        3. The first number is 3 more than twice the second number:
           x = 2y + 3
        Final Equations: 
           x + y = 12 and x = 2y + 3
        """
    },
    {
        "problem": "The sum of two numbers is 22. The second number is 4 less than the first number. Write a system of linear equations to represent this information and find the values of the two numbers.",
        "reasoning": """
        1. Let x be the first number and y be the second number.
        2. The sum of the two numbers is:
           x + y = 22
        3. The second number is 4 less than the first number:
           y = x - 4
        Final Equations: 
           x + y = 22 and y = x - 4
        """
    },
    {
        "problem": "A train travels 100 miles downstream in 2 hours and 60 miles upstream in 3 hours. Write a system of linear equations to represent this information and find the speed of the train in still water and the speed of the current.",
        "reasoning": """
        1. Let x be the speed of the train in still water and y be the speed of the current.
        2. The effective speed downstream is (x + y):
           2(x + y) = 100
        3. The effective speed upstream is (x - y):
           3(x - y) = 60
        Final Equations: 
           2(x + y) = 100 and 3(x - y) = 60
        """
    },
    {
        "problem": "A person works at a rate of 1/5 of the job per day. Another person works at a rate of 1/4 of the job per day. Write a system of linear equations to represent the time it takes for them to finish the job together.",
        "reasoning": """
        1. Let x be the number of days for the first person to finish the job alone, and y be the number of days for the second person.
        2. The rate of work is 1/5 per day for the first person:
           1/x = 1/5
        3. The rate of work is 1/4 per day for the second person:
           1/y = 1/4
        Final Equations: 
           1/x = 1/5 and 1/y = 1/4
        """
    },
    {
        "problem": "The sum of three numbers is 30. The first number is 4 more than the second number, and the third number is 5 less than the first number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The sum of the three numbers is:
           x + y + z = 30
        3. The first number is 4 more than the second number:
           x = y + 4
        4. The third number is 5 less than the first number:
           z = x - 5
        Final Equations: 
           x + y + z = 30, x = y + 4, and z = x - 5
        """
    },
    {
        "problem": "The sum of three numbers is 50. The first number is twice the second number, and the third number is 3 less than the first number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The sum of the three numbers is:
           x + y + z = 50
        3. The first number is twice the second number:
           x = 2y
        4. The third number is 3 less than the first number:
           z = x - 3
        Final Equations: 
           x + y + z = 50, x = 2y, and z = x - 3
        """
    },
    {
        "problem": "A company sells three types of products: A, B, and C. The total revenue from selling 2 units of A, 3 units of B, and 5 units of C is $40. The total revenue from selling 3 units of A, 4 units of B, and 7 units of C is $50. The total revenue from selling 4 units of A, 5 units of B, and 8 units of C is $60. Write a system of linear equations to represent this information and find the price of each product.",
        "reasoning": """
        1. Let x be the price of product A, y be the price of product B, and z be the price of product C.
        2. The first revenue equation is:
           2x + 3y + 5z = 40
        3. The second revenue equation is:
           3x + 4y + 7z = 50
        4. The third revenue equation is:
           4x + 5y + 8z = 60
        Final Equations: 
           2x + 3y + 5z = 40, 3x + 4y + 7z = 50, and 4x + 5y + 8z = 60
        """
    },
    {
        "problem": "A person invests $1000 in three different accounts. The first account earns 5% interest, the second earns 6%, and the third earns 7%. The total interest earned in one year is $55. Write a system of linear equations to represent this information and find the amount invested in each account.",
        "reasoning": """
        1. Let x be the amount invested in the first account, y be the amount invested in the second account, and z be the amount invested in the third account.
        2. The total amount invested is:
           x + y + z = 1000
        3. The total interest earned from the investments is:
           0.05x + 0.06y + 0.07z = 55
        Final Equations: 
           x + y + z = 1000 and 0.05x + 0.06y + 0.07z = 55
        """
    },
    {
        "problem": "The sum of three numbers is 72. The first number is 3 times the second number, and the third number is 4 less than the first number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The sum of the three numbers is:
           x + y + z = 72
        3. The first number is 3 times the second number:
           x = 3y
        4. The third number is 4 less than the first number:
           z = x - 4
        Final Equations: 
           x + y + z = 72, x = 3y, and z = x - 4
        """
    },
    {
        "problem": "A total of 100 animals are in a zoo, consisting of birds, lions, and tigers. The number of birds is 20 more than the number of lions. The number of tigers is twice the number of lions. Write a system of linear equations to represent this information and find the number of each animal.",
        "reasoning": """
        1. Let x be the number of birds, y be the number of lions, and z be the number of tigers.
        2. The total number of animals is:
           x + y + z = 100
        3. The number of birds is 20 more than the number of lions:
           x = y + 20
        4. The number of tigers is twice the number of lions:
           z = 2y
        Final Equations: 
           x + y + z = 100, x = y + 20, and z = 2y
        """
    },
    {
        "problem": "Three numbers add up to 48. The first number is twice the second number, and the second number is 3 less than the third number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The sum of the three numbers is:
           x + y + z = 48
        3. The first number is twice the second number:
           x = 2y
        4. The second number is 3 less than the third number:
           y = z - 3
        Final Equations: 
           x + y + z = 48, x = 2y, and y = z - 3
        """
    },
    {
        "problem": "A company produces three types of shirts: red, blue, and green. The total number of shirts produced is 500. The number of red shirts is twice the number of blue shirts, and the number of green shirts is 50 more than the number of blue shirts. Write a system of linear equations to represent this information and find the number of each type of shirt produced.",
        "reasoning": """
        1. Let x be the number of red shirts, y be the number of blue shirts, and z be the number of green shirts.
        2. The total number of shirts is:
           x + y + z = 500
        3. The number of red shirts is twice the number of blue shirts:
           x = 2y
        4. The number of green shirts is 50 more than the number of blue shirts:
           z = y + 50
        Final Equations: 
           x + y + z = 500, x = 2y, and z = y + 50
        """
    },
    {
        "problem": "The sum of three numbers is 80. The first number is 10 more than twice the second number, and the third number is 5 less than the second number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The sum of the three numbers is:
           x + y + z = 80
        3. The first number is 10 more than twice the second number:
           x = 2y + 10
        4. The third number is 5 less than the second number:
           z = y - 5
        Final Equations: 
           x + y + z = 80, x = 2y + 10, and z = y - 5
        """
    },
    {
        "problem": "A store sells 3 types of fruits: apples, bananas, and cherries. The total number of fruits sold is 120. The number of apples is twice the number of bananas, and the number of cherries is 10 less than the number of apples. Write a system of linear equations to represent this information and find the number of each type of fruit sold.",
        "reasoning": """
        1. Let x be the number of apples, y be the number of bananas, and z be the number of cherries.
        2. The total number of fruits sold is:
           x + y + z = 120
        3. The number of apples is twice the number of bananas:
           x = 2y
        4. The number of cherries is 10 less than the number of apples:
           z = x - 10
        Final Equations: 
           x + y + z = 120, x = 2y, and z = x - 10
        """
    },
    {
        "problem": "Three friends decided to pool their money together for a trip. The first friend contributed $50 more than the second friend, and the second friend contributed $30 more than the third friend. The total amount of money contributed is $330. Write a system of linear equations to represent this information and find the amount each friend contributed.",
        "reasoning": """
        1. Let x be the amount contributed by the first friend, y be the amount contributed by the second friend, and z be the amount contributed by the third friend.
        2. The total amount contributed is:
           x + y + z = 330
        3. The first friend contributed $50 more than the second friend:
           x = y + 50
        4. The second friend contributed $30 more than the third friend:
           y = z + 30
        Final Equations: 
           x + y + z = 330, x = y + 50, and y = z + 30
        """
    },
    {
        "problem": "The sum of three numbers is 72. The first number is 2 times the second number, and the third number is 8 more than the second number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The sum of the three numbers is:
           x + y + z = 72
        3. The first number is 2 times the second number:
           x = 2y
        4. The third number is 8 more than the second number:
           z = y + 8
        Final Equations: 
           x + y + z = 72, x = 2y, and z = y + 8
        """
    },
    {
        "problem": "Three numbers add up to 90. The first number is 4 times the second number, and the second number is 5 less than the third number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The sum of the three numbers is:
           x + y + z = 90
        3. The first number is 4 times the second number:
           x = 4y
        4. The second number is 5 less than the third number:
           y = z - 5
        Final Equations: 
           x + y + z = 90, x = 4y, and y = z - 5
        """
    },
    {
        "problem": "A class has 40 students. The number of boys is twice the number of girls, and the number of girls is 10 less than the number of boys. Write a system of linear equations to represent this information and find the number of boys and girls in the class.",
        "reasoning": """
        1. Let x be the number of boys and y be the number of girls.
        2. The total number of students is:
           x + y = 40
        3. The number of boys is twice the number of girls:
           x = 2y
        4. The number of girls is 10 less than the number of boys:
           y = x - 10
        Final Equations: 
           x + y = 40, x = 2y, and y = x - 10
        """
    },
    {
        "problem": "Three types of fruit are sold: apples, bananas, and oranges. The total number of fruits sold is 160. The number of apples is twice the number of bananas, and the number of bananas is 10 less than the number of oranges. Write a system of linear equations to represent this information and find the number of each type of fruit sold.",
        "reasoning": """
        1. Let x be the number of apples, y be the number of bananas, and z be the number of oranges.
        2. The total number of fruits sold is:
           x + y + z = 160
        3. The number of apples is twice the number of bananas:
           x = 2y
        4. The number of bananas is 10 less than the number of oranges:
           y = z - 10
        Final Equations: 
           x + y + z = 160, x = 2y, and y = z - 10
        """
    },
    {
        "problem": "The sum of three numbers is 120. The first number is 5 more than the second number, and the third number is twice the second number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The sum of the three numbers is:
           x + y + z = 120
        3. The first number is 5 more than the second number:
           x = y + 5
        4. The third number is twice the second number:
           z = 2y
        Final Equations: 
           x + y + z = 120, x = y + 5, and z = 2y
        """
    },
    {
        "problem": "The sum of three numbers is 60. The first number is 5 more than the second number, and the second number is 5 less than the third number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The sum of the three numbers is:
           x + y + z = 60
        3. The first number is 5 more than the second number:
           x = y + 5
        4. The second number is 5 less than the third number:
           y = z - 5
        Final Equations: 
           x + y + z = 60, x = y + 5, and y = z - 5
        """
    },
    {
        "problem": "The sum of three numbers is 100. The first number is 3 times the second number, and the second number is 5 less than the third number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The sum of the three numbers is:
           x + y + z = 100
        3. The first number is 3 times the second number:
           x = 3y
        4. The second number is 5 less than the third number:
           y = z - 5
        Final Equations: 
           x + y + z = 100, x = 3y, and y = z - 5
        """
    },
    {
        "problem": "The sum of three numbers is 150. The first number is 4 times the second number, and the third number is 15 more than the second number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The sum of the three numbers is:
           x + y + z = 150
        3. The first number is 4 times the second number:
           x = 4y
        4. The third number is 15 more than the second number:
           z = y + 15
        Final Equations: 
           x + y + z = 150, x = 4y, and z = y + 15
        """
    },
    {
        "problem": "The total number of students in a school is 300. The number of boys is 100 more than the number of girls, and the number of girls is 50 less than the number of teachers. Write a system of linear equations to represent this information and find the number of boys, girls, and teachers.",
        "reasoning": """
        1. Let x be the number of boys, y be the number of girls, and z be the number of teachers.
        2. The total number of students is:
           x + y = 300
        3. The number of boys is 100 more than the number of girls:
           x = y + 100
        4. The number of girls is 50 less than the number of teachers:
           y = z - 50
        Final Equations: 
           x + y = 300, x = y + 100, and y = z - 50
        """
    },
    {
        "problem": "The sum of three numbers is 45. The first number is 3 times the second number, and the third number is 10 less than the second number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The sum of the three numbers is:
           x + y + z = 45
        3. The first number is 3 times the second number:
           x = 3y
        4. The third number is 10 less than the second number:
           z = y - 10
        Final Equations: 
           x + y + z = 45, x = 3y, and z = y - 10
        """
    },
    {
        "problem": "The total of three numbers is 200. The first number is twice the second number, and the second number is 30 less than the third number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The total of the three numbers is:
           x + y + z = 200
        3. The first number is twice the second number:
           x = 2y
        4. The second number is 30 less than the third number:
           y = z - 30
        Final Equations: 
           x + y + z = 200, x = 2y, and y = z - 30
        """
    },
    {
        "problem": "A box contains apples, bananas, and oranges. The total number of fruits is 180. The number of apples is 50 more than the number of bananas, and the number of bananas is 20 less than the number of oranges. Write a system of linear equations to represent this information and find the number of each type of fruit.",
        "reasoning": """
        1. Let x be the number of apples, y be the number of bananas, and z be the number of oranges.
        2. The total number of fruits is:
           x + y + z = 180
        3. The number of apples is 50 more than the number of bananas:
           x = y + 50
        4. The number of bananas is 20 less than the number of oranges:
           y = z - 20
        Final Equations: 
           x + y + z = 180, x = y + 50, and y = z - 20
        """
    },
    {
        "problem": "The sum of three numbers is 50. The first number is 4 times the second number, and the third number is 10 more than the second number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The sum of the three numbers is:
           x + y + z = 50
        3. The first number is 4 times the second number:
           x = 4y
        4. The third number is 10 more than the second number:
           z = y + 10
        Final Equations: 
           x + y + z = 50, x = 4y, and z = y + 10
        """
    },
    {
        "problem": "A farm has cows, chickens, and goats. The total number of animals is 120. The number of cows is 10 more than twice the number of chickens, and the number of goats is 20 more than the number of chickens. Write a system of linear equations to represent this information and find the number of each type of animal.",
        "reasoning": """
        1. Let x be the number of cows, y be the number of chickens, and z be the number of goats.
        2. The total number of animals is:
           x + y + z = 120
        3. The number of cows is 10 more than twice the number of chickens:
           x = 2y + 10
        4. The number of goats is 20 more than the number of chickens:
           z = y + 20
        Final Equations: 
           x + y + z = 120, x = 2y + 10, and z = y + 20
        """
    },
    {
        "problem": "A family has 1500 dollars. The father has 3 times as much as the mother, and the mother has 200 dollars more than the son. Write a system of linear equations to represent this information and find the amount of money each family member has.",
        "reasoning": """
        1. Let x be the amount of money the father has, y be the amount of money the mother has, and z be the amount of money the son has.
        2. The total amount of money is:
           x + y + z = 1500
        3. The father has 3 times as much as the mother:
           x = 3y
        4. The mother has 200 dollars more than the son:
           y = z + 200
        Final Equations: 
           x + y + z = 1500, x = 3y, and y = z + 200
        """
    },
    {
        "problem": "Three students decided to contribute to a class fund. The first student contributed $20 more than the second student, and the second student contributed $10 more than the third student. The total contribution was $90. Write a system of linear equations to represent this information and find the amount each student contributed.",
        "reasoning": """
        1. Let x be the amount the first student contributed, y be the amount the second student contributed, and z be the amount the third student contributed.
        2. The total contribution is:
           x + y + z = 90
        3. The first student contributed $20 more than the second student:
           x = y + 20
        4. The second student contributed $10 more than the third student:
           y = z + 10
        Final Equations: 
           x + y + z = 90, x = y + 20, and y = z + 10
        """
    },
    {
        "problem": "The sum of three numbers is 120. The first number is 5 times the second number, and the second number is 10 less than the third number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The sum of the three numbers is:
           x + y + z = 120
        3. The first number is 5 times the second number:
           x = 5y
        4. The second number is 10 less than the third number:
           y = z - 10
        Final Equations: 
           x + y + z = 120, x = 5y, and y = z - 10
        """
    },
    {
        "problem": "The sum of three numbers is 240. The first number is 6 times the second number, and the second number is 15 less than the third number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The sum of the three numbers is:
           x + y + z = 240
        3. The first number is 6 times the second number:
           x = 6y
        4. The second number is 15 less than the third number:
           y = z - 15
        Final Equations: 
           x + y + z = 240, x = 6y, and y = z - 15
        """
    },
    {
        "problem": "A factory produces pencils, pens, and erasers. The total number of items produced is 1000. The number of pencils is 3 times the number of pens, and the number of pens is 100 more than the number of erasers. Write a system of linear equations to represent this information and find the number of each item produced.",
        "reasoning": """
        1. Let x be the number of pencils, y be the number of pens, and z be the number of erasers.
        2. The total number of items produced is:
           x + y + z = 1000
        3. The number of pencils is 3 times the number of pens:
           x = 3y
        4. The number of pens is 100 more than the number of erasers:
           y = z + 100
        Final Equations: 
           x + y + z = 1000, x = 3y, and y = z + 100
        """
    },
    {
        "problem": "Three people contributed money to a charity. The first person contributed $10 more than the second person, and the second person contributed $5 more than the third person. The total contribution was $80. Write a system of linear equations to represent this information and find the amount each person contributed.",
        "reasoning": """
        1. Let x be the amount the first person contributed, y be the amount the second person contributed, and z be the amount the third person contributed.
        2. The total contribution is:
           x + y + z = 80
        3. The first person contributed $10 more than the second person:
           x = y + 10
        4. The second person contributed $5 more than the third person:
           y = z + 5
        Final Equations: 
           x + y + z = 80, x = y + 10, and y = z + 5
        """
    },
    {
        "problem": "The sum of three numbers is 50. The first number is twice the second number, and the second number is 5 less than the third number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The sum of the three numbers is:
           x + y + z = 50
        3. The first number is twice the second number:
           x = 2y
        4. The second number is 5 less than the third number:
           y = z - 5
        Final Equations: 
           x + y + z = 50, x = 2y, and y = z - 5
        """
    },
    {
        "problem": "The sum of three numbers is 120. The first number is 3 times the second number, and the third number is 10 less than the second number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The sum of the three numbers is:
           x + y + z = 120
        3. The first number is 3 times the second number:
           x = 3y
        4. The third number is 10 less than the second number:
           z = y - 10
        Final Equations: 
           x + y + z = 120, x = 3y, and z = y - 10
        """
    },
    {
        "problem": "A person has a total of 500 dollars. The amount of money they have is split between three accounts. The first account contains 3 times the amount of the second account, and the second account contains 50 dollars more than the third account. Write a system of linear equations to represent this information and find the amount in each account.",
        "reasoning": """
        1. Let x be the amount in the first account, y be the amount in the second account, and z be the amount in the third account.
        2. The total amount is:
           x + y + z = 500
        3. The first account contains 3 times the amount of the second account:
           x = 3y
        4. The second account contains 50 dollars more than the third account:
           y = z + 50
        Final Equations: 
           x + y + z = 500, x = 3y, and y = z + 50
        """
    },
    {
        "problem": "The sum of three numbers is 150. The first number is 4 times the second number, and the third number is 20 more than the second number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The sum of the three numbers is:
           x + y + z = 150
        3. The first number is 4 times the second number:
           x = 4y
        4. The third number is 20 more than the second number:
           z = y + 20
        Final Equations: 
           x + y + z = 150, x = 4y, and z = y + 20
        """
    },
    {
        "problem": "The sum of three numbers is 300. The first number is twice the second number, and the third number is 50 less than the second number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The sum of the three numbers is:
           x + y + z = 300
        3. The first number is twice the second number:
           x = 2y
        4. The third number is 50 less than the second number:
           z = y - 50
        Final Equations: 
           x + y + z = 300, x = 2y, and z = y - 50
        """
    },
    {
        "problem": "The total number of books in a library is 180. The number of fiction books is twice the number of non-fiction books, and the number of non-fiction books is 20 less than the number of reference books. Write a system of linear equations to represent this information and find the number of each type of book.",
        "reasoning": """
        1. Let x be the number of fiction books, y be the number of non-fiction books, and z be the number of reference books.
        2. The total number of books is:
           x + y + z = 180
        3. The number of fiction books is twice the number of non-fiction books:
           x = 2y
        4. The number of non-fiction books is 20 less than the number of reference books:
           y = z - 20
        Final Equations: 
           x + y + z = 180, x = 2y, and y = z - 20
        """
    },
    {
        "problem": "The sum of three numbers is 500. The first number is 5 times the second number, and the third number is 20 more than the second number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The sum of the three numbers is:
           x + y + z = 500
        3. The first number is 5 times the second number:
           x = 5y
        4. The third number is 20 more than the second number:
           z = y + 20
        Final Equations: 
           x + y + z = 500, x = 5y, and z = y + 20
        """
    },
    {
        "problem": "The sum of three numbers is 250. The first number is 8 times the second number, and the second number is 30 less than the third number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The sum of the three numbers is:
           x + y + z = 250
        3. The first number is 8 times the second number:
           x = 8y
        4. The second number is 30 less than the third number:
           y = z - 30
        Final Equations: 
           x + y + z = 250, x = 8y, and y = z - 30
        """
    },
    {
        "problem": "A company has three types of products: A, B, and C. The total number of products produced is 600. The number of product A is 4 times the number of product B, and the number of product B is 50 more than the number of product C. Write a system of linear equations to represent this information and find the number of each product produced.",
        "reasoning": """
        1. Let x be the number of product A, y be the number of product B, and z be the number of product C.
        2. The total number of products is:
           x + y + z = 600
        3. The number of product A is 4 times the number of product B:
           x = 4y
        4. The number of product B is 50 more than the number of product C:
           y = z + 50
        Final Equations: 
           x + y + z = 600, x = 4y, and y = z + 50
        """
    },
    {
        "problem": "The sum of three numbers is 150. The first number is 3 times the second number, and the third number is 15 more than the second number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The sum of the three numbers is:
           x + y + z = 150
        3. The first number is 3 times the second number:
           x = 3y
        4. The third number is 15 more than the second number:
           z = y + 15
        Final Equations: 
           x + y + z = 150, x = 3y, and z = y + 15
        """
    },
    {
        "problem": "The sum of three numbers is 400. The first number is 7 times the second number, and the second number is 40 less than the third number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The sum of the three numbers is:
           x + y + z = 400
        3. The first number is 7 times the second number:
           x = 7y
        4. The second number is 40 less than the third number:
           y = z - 40
        Final Equations: 
           x + y + z = 400, x = 7y, and y = z - 40
        """
    },
    {
        "problem": "A store sells three types of fruits: apples, bananas, and cherries. The total number of fruits sold is 200. The number of apples is twice the number of bananas, and the number of bananas is 30 more than the number of cherries. Write a system of linear equations to represent this information and find the number of each type of fruit sold.",
        "reasoning": """
        1. Let x be the number of apples, y be the number of bananas, and z be the number of cherries.
        2. The total number of fruits sold is:
           x + y + z = 200
        3. The number of apples is twice the number of bananas:
           x = 2y
        4. The number of bananas is 30 more than the number of cherries:
           y = z + 30
        Final Equations: 
           x + y + z = 200, x = 2y, and y = z + 30
        """
    },
    {
        "problem": "The total number of seats in a theater is 800. The number of front-row seats is 5 times the number of middle-row seats, and the number of middle-row seats is 100 more than the number of back-row seats. Write a system of linear equations to represent this information and find the number of seats in each row.",
        "reasoning": """
        1. Let x be the number of front-row seats, y be the number of middle-row seats, and z be the number of back-row seats.
        2. The total number of seats is:
           x + y + z = 800
        3. The number of front-row seats is 5 times the number of middle-row seats:
           x = 5y
        4. The number of middle-row seats is 100 more than the number of back-row seats:
           y = z + 100
        Final Equations: 
           x + y + z = 800, x = 5y, and y = z + 100
        """
    },
    {
        "problem": "The sum of three numbers is 360. The first number is 6 times the second number, and the third number is 30 less than the second number. Write a system of linear equations to represent this information and find the values of the three numbers.",
        "reasoning": """
        1. Let x be the first number, y be the second number, and z be the third number.
        2. The sum of the three numbers is:
           x + y + z = 360
        3. The first number is 6 times the second number:
           x = 6y
        4. The third number is 30 less than the second number:
           z = y - 30
        Final Equations: 
           x + y + z = 360, x = 6y, and z = y - 30
        """
    },
    {
        "problem": "The total number of chairs in a conference room is 250. The number of chairs in the front row is twice the number of chairs in the middle row, and the number of chairs in the middle row is 30 more than the number of chairs in the back row. Write a system of linear equations to represent this information and find the number of chairs in each row.",
        "reasoning": """
        1. Let x be the number of chairs in the front row, y be the number of chairs in the middle row, and z be the number of chairs in the back row.
        2. The total number of chairs is:
           x + y + z = 250
        3. The number of chairs in the front row is twice the number of chairs in the middle row:
           x = 2y
        4. The number of chairs in the middle row is 30 more than the number of chairs in the back row:
           y = z + 30
        Final Equations: 
           x + y + z = 250, x = 2y, and y = z + 30
        """
    },
    {
        "problem": "A person invested in two schemes, one offering 6% annual interest and the other offering 8% annual interest. The total investment was $10,000, and the total interest earned from both schemes after one year was $720. Write a system of linear equations to represent this information and find how much was invested in each scheme.",
        "reasoning": """
        1. Let x be the amount invested at 6% interest, and y be the amount invested at 8% interest.
        2. The total investment is:
           x + y = 10000
        3. The total interest is $720:
           0.06x + 0.08y = 720
        Final Equations:
           x + y = 10000 and 0.06x + 0.08y = 720
        """
    },
    {
        "problem": "A car rental company offers two types of cars. One type rents for $30 per day, and the other rents for $45 per day. If a customer rents the cars for a total of 6 days and the total cost is $240, write a system of linear equations to represent this information and find how many days the customer rented each type of car.",
        "reasoning": """
        1. Let x be the number of days the customer rented the $30 per day car, and y be the number of days they rented the $45 per day car.
        2. The total days rented is:
           x + y = 6
        3. The total cost is $240:
           30x + 45y = 240
        Final Equations:
           x + y = 6 and 30x + 45y = 240
        """
    },
    {
        "problem": "A farmer has a total of 1000 meters of fencing. She wants to use the fencing to enclose a rectangular area for grazing. The length of the enclosure is 3 meters longer than twice the width. Write a system of linear equations to represent this information and find the dimensions of the enclosure.",
        "reasoning": """
        1. Let x be the width of the enclosure, and y be the length of the enclosure.
        2. The total fencing is used for the perimeter:
           2x + 2y = 1000
        3. The length is 3 meters longer than twice the width:
           y = 2x + 3
        Final Equations:
           2x + 2y = 1000 and y = 2x + 3
        """
    },
    {
        "problem": "Two types of tickets are sold for a concert: adult tickets and child tickets. The price of an adult ticket is $25, and the price of a child ticket is $10. If 100 tickets are sold and the total revenue is $2,000, write a system of linear equations to represent this information and find how many adult and child tickets were sold.",
        "reasoning": """
        1. Let x be the number of adult tickets sold, and y be the number of child tickets sold.
        2. The total number of tickets sold is:
           x + y = 100
        3. The total revenue is $2,000:
           25x + 10y = 2000
        Final Equations:
           x + y = 100 and 25x + 10y = 2000
        """
    },
    {
        "problem": "A train travels from city A to city B and then returns. The time it takes to travel from A to B is 2 hours less than the time it takes to travel from B to A. The total travel time for the round trip is 8 hours. Write a system of linear equations to represent this information and find the time taken for each leg of the journey.",
        "reasoning": """
        1. Let x be the time taken to travel from A to B, and y be the time taken to travel from B to A.
        2. The total travel time is:
           x + y = 8
        3. The time from A to B is 2 hours less than the time from B to A:
           x = y - 2
        Final Equations:
           x + y = 8 and x = y - 2
        """
    },
    {
        "problem": "A boat travels 120 kilometers downstream in 3 hours and returns upstream in 4 hours. The speed of the current is 2 km/h. Write a system of linear equations to represent this information and find the speed of the boat in still water.",
        "reasoning": """
        1. Let x be the speed of the boat in still water (in km/h), and y be the speed of the current (which is given as 2 km/h).
        2. The speed of the boat downstream is (x + y), and it takes 3 hours to travel 120 km:
           120 = 3(x + 2)
        3. The speed of the boat upstream is (x - y), and it takes 4 hours to travel 120 km:
           120 = 4(x - 2)
        Final Equations:
           120 = 3(x + 2) and 120 = 4(x - 2)
        """
    }



]