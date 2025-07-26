test_cases = [
    {
        "input": "The sum of two numbers is 18, and their difference is 4. Find the numbers.",
        "expected_equation": "x + y = 18 and x - y = 4",
        "weight": 4
    },
    {
        "input": "A person is twice as old as his friend. Five years ago, the sum of their ages was 30. Find their current ages.",
        "expected_equation": "x = 2 * y and x - 5 + y - 5 = 30",
        "weight": 5
    },
    {
        "input": "A car rental company charges a fixed fee of $50 and an additional $3 per mile. If the total cost for a trip is $110, how many miles were driven?",
        "expected_equation": "50 + 3 * x = 110",
        "weight": 5
    },
    {
        "input": "The perimeter of a rectangle is 24 meters, and the length is twice the width. Find the dimensions of the rectangle.",
        "expected_equation": "2 * (x + y) = 24 and x = 2 * y",
        "weight": 5
    },
    {
        "input": "In a class, the ratio of boys to girls is 3:2. If there are 40 students in total, how many boys and girls are there?",
        "expected_equation": "x + y = 40 and x/y = 3/2",
        "weight": 6
    },
    {
        "input": "If 4 pens and 3 notebooks cost $32, and 3 pens and 4 notebooks cost $31, find the cost of each pen and each notebook.",
        "expected_equation": "4 * x + 3 * y = 32 and 3 * x + 4 * y = 31",
        "weight": 7
    },
    {
        "input": "A group of friends bought 3 sandwiches and 2 drinks for $22, and then later bought 4 sandwiches and 3 drinks for $30. Find the price of each sandwich and each drink.",
        "expected_equation": "3 * x + 2 * y = 22 and 4 * x + 3 * y = 30",
        "weight": 7
    },
    {
        "input": "A father is 3 times as old as his son. In 5 years, the sum of their ages will be 70. Find their current ages.",
        "expected_equation": "x = 3 * y and x + 5 + y + 5 = 70",
        "weight": 7
    },
    {
        "input": "The sum of three numbers is 20. The second number is twice the first, and the third number is three times the first. Find the numbers.",
        "expected_equation": "x + y + z = 20, y = 2x, z = 3x",
        "weight": 8
    },
    {
        "input": "A shop sells two types of items. Three of item A and two of item B cost $17, while five of item A and three of item B cost $27. Find the price of each item.",
        "expected_equation": "3 * x + 2 * y = 17 and 5 * x + 3 * y = 27",
        "weight": 8
    },
    {
        "input": "A chemist has two solutions, one containing 40% alcohol and the other containing 60% alcohol. How many liters of each should be mixed to obtain 30 liters of a 50% alcohol solution?",
        "expected_equation": "0.4 * x + 0.6 * y = 0.5 * 30 and x + y = 30",
        "weight": 8
    },
    {
        "input": "A vendor sells three items: apples, oranges, and bananas. One apple, one orange, and one banana cost $6. Two apples, one oranges, and three bananas cost $13. Three apples, four oranges, and one banana cost $14. Find the price of each fruit.",
        "expected_equation": " x +  y + z = 6 and 2* x +  y + 3* z =13 and 3* x + 4 * y + z = 14",
        "weight": 10
    },
    {
        "input": "In a school, the number of boys is twice the number of girls. If 10 new boys and 15 new girls join, the ratio becomes 5:3. Find the initial number of boys and girls.",
        "expected_equation": "x = 2 * y and (x + 10)/(y + 15) = 5/3",
        "weight": 9
    },
    {
        "input": "Two trains leave two stations 400 miles apart and travel toward each other. Train A travels at 60 mph, and Train B at 40 mph. How long will it take for them to meet?",
        "expected_equation": "60 * x + 40 * x = 400",
        "weight": 5
    },
    {
        "input": "A mixture contains milk and water in the ratio 5:4. If 15 liters of water are added, the ratio becomes 1:1. Find the initial quantities of milk and water.",
        "expected_equation": "x/y = 5/4 and x/(y + 15) = 1",
        "weight": 7
    },
    {
        "input": "A bookstore sold 4 fiction books and 3 non-fiction books for $80. The next day, they sold 5 fiction books and 2 non-fiction books for $70. Find the price of each type of book.",
        "expected_equation": "4 * x + 3 * y = 80 and 5 * x + 2 * y = 70",
        "weight": 6
    },
    {
        "input": "A car travels at a speed of 50 mph. How long will it take to travel 120 miles?",
        "expected_equation": "50 * x = 120",
        "weight": 3
    },

    {
    "input": "A boat travels 30 km downstream in 2 hours and takes 5 hours to travel the same distance upstream. Find the speed of the boat in still water and the speed of the current.",
    "expected_equation": "(30/(x + y)) = 2, (30/(x - y)) = 5",
    "weight": 4
    },

    {
        "input": "The sum of three consecutive integers is 24. Find the integers.",
        "expected_equation": "x + (x + 1) + (x + 2) = 24",
        "weight": 4
    },
    {
        "input": "In a survey, 60% of people preferred coffee over tea. If 240 people were surveyed, how many preferred tea?",
        "expected_equation": "0.6 * x + 0.4 * x = 240",
        "weight": 3
    }

]
