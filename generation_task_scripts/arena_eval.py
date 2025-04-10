
examples = [
    "Local Charity Raises Record-Breaking Funds at Weekend Fun Run",
    "Historic Library Reopens After Major Renovations",
    "Groundbreaking Study Reveals Surprising Benefits of Afternoon Naps",
    "Neighborhood Bakery Launches Viral 'Secret Recipe' Dessert Challenge",
    "Tech Startup Revolutionizes Home Security With Smart Drone System",
    "Archaeologists Unearth Hidden Chamber in Centuries-Old Temple",
    "Promising Young Musician Secures Streaming Sensation With Debut Single",
    "Innovative Urban Farm Brings Fresh Produce to Underserved Communities",
    "Wildlife Experts Celebrate Return of Endangered Species to National Park",
    "Local High School Students Win International Robotics Championship",
    "Hometown Chef's Fusion Cuisine Impresses Judges on Culinary TV Show",
    "Researchers Propose New Method to Clean Polluted Rivers Using Algae",
    "Pop-Up Shop Showcases Sustainable Fashion Trends for Eco-Conscious Shoppers",
    "Global Survey Reveals Shift in Work-Life Balance Priorities After Pandemic",
    "Hacker Group Claims to Expose Major Security Flaws in Popular Messaging App",
    "Rising Actor Lands Breakthrough Role in Acclaimed Historical Drama",
    "City Council Votes to Expand Public Green Spaces, Plant Over 1,000 Trees",
    "Gaming World Buzzes Over Rumored Console Release and Innovative Features",
    "Scientists Develop Biodegradable Plastics That Decompose in 90 Days",
    "Teenager Invents Portable Water Purifier to Aid Communities in Crisis",
    "Museum Launches Virtual Reality Exhibit to Bring Ancient Civilizations to Life",
    "Community Theater Debuts Vibrant Take on Classic Shakespearean Comedy",
    "Astronomers Detect Mysterious Signals From Newly Discovered Exoplanet",
    "Marathon Runner Overcomes Injury to Claim Victory in Stunning Comeback",
    "Brewery Collaborates With Local Artists to Create Limited-Edition Labels",
    "Dentists Nationwide Advocate for Inclusive Care for Patients With Disabilities",
    "Documentary Filmmaker Sheds Light on Unseen Effects of Climate Change",
    "Online Learning Platform Expands Academic Reach to Rural Communities",
    "Neighborhood Restoration Effort Transforms Abandoned Lots Into Community Gardens",
    "Award-Winning Architect Designs Futuristic Skyscraper With Zero Carbon Footprint",
]

answers_1 = [
    "Charity has fun, raises money. Who knew?",
    "Major renovations, minor improvements.",
    "Groundbreaking study finds naps can make you feel like less of a functioning member of society.",
    "Bakery creates a challenge? Oh, the humanity!",
    "Tech startup's 'smart drone system' can now watch you—in the privacy of your own home. Great, right?",
    "Hey, who knew? Maybe archaeologists will find something interesting eventually.",
    "Sensation! This Debut Single is a Musical Sensation. That's Absolutely Not a Clickbait Headline.",
    "Farm in the hood? That's news?",
    "Wildlife experts: the new cool thing to be on Instagram.",
    "Local High School Students Win International Robotics Championship? Wow, The Teachers Have To Lift Their Heads Off Their Desks.",
    "Hometown chef serves fusion food on TV and it's not bad, shocking no one.",
    "Algae. It's not just for breakfast anymore.",
    "Pop-Up Shop Showcases Sustainable Fashion Trends for Eco-Conscious Shoppers. Finally, a reason to shop.",
    "Priorities? Oh, you mean the thing you don't have.",
    "Another day, another security flaw - how unique.",
    "Breakthrough role lands actor.",
    "City council votes to grow green spaces, but not greed.",
    "Gaming world: still buzzing over a rumor.",
    "Biodegradable plastic. It's like the ultimate Instagram challenge.",
    "Teenager invents portable water purifier to aid communities in crisis. Oh, great—now we just need an air-conditioned school bus full of vaccines and",
    "Museum adds virtual reality to exhibits. Innovation ensues.",
    "Community theater: the best comedy you never saw coming",
    "Astronomers detect weird signals. Now they just need to explain why the aliens are texting them.",
    "Runner Overcomes Injury to Claim Victory",
    "Brewery gets artsy. Now you can drink an art show.",
    "Dentists want inclusivity, but only when they're not the ones with teeth to fix.",
    "Filmmaker: Hey, doomsday is sexy now!",
    "Online learning platform takes its magic show on the road.",
    "Neighborhood Restoration Effort Turns Abandoned Lots into Community Gardens",
    "Award-winning architect builds a building that will probably never exist.",
]

answers_2 = [
    "Local Charity Somehow Convinces People to Pay for the Privilege of Running",
    "Historic Library Finally Reopens After Years of Construction Noise and Delays",
    "Scientists Shocked to Learn That Sleep Is, In Fact, Good for You",
    "Neighborhood Bakery Hopes Desperate Trend-Chasers Will Overpay for Mystery Pastry",
    "Tech Startup Promises Your Expensive New Gadget Will Spy on You More Efficiently",
    "Archaeologists Find Yet Another Room Full of Dusty Old Things",
    "Another Teenager Becomes Overnight Millionaire by Making Sounds Into a Microphone",
    "Urban Farm Heroically Grows Lettuce in a Place Where Lettuce Normally Doesn't Grow",
    "Endangered Species Returns Just in Time to Be Displaced by New Tourist Resort",
    "Local Teens Prove They're Smarter Than the Adults Who Underfund Their Schools",
    "Hometown Chef Finally Gets 15 Minutes of Fame Before the Next One Does",
    "Researchers Suggest Using Algae to Clean Rivers—Because What Could Go Wrong?",
    "Pop-Up Shop Sells $200 T-Shirts to People Who Want to Feel Better About Consumerism",
    "Global Survey Confirms That People Like Not Being Overworked—Shocking",
    "Hackers Expose Security Flaws That Company Will Ignore Until It's Too Late",
    "Another Attractive Person Lands Role Where They Wear Old-Timey Clothes",
    "City Council Does Bare Minimum to Combat Concrete Jungle They Created",
    "Gamers Lose Their Minds Over Slightly Shinier Version of Last Year's Console",
    "Scientists Invent Plastic That Disappears—Just in Time for Regular Plastic to Be Everywhere Forever",
    "Teenager Solves Water Crisis That Governments Have Ignored for Decades",
    "Museum Uses Expensive Tech to Show You What Books Could've Told You for Free",
    "Community Theater Attempts Shakespeare—Again",
    "Astronomers Detect Mysterious Signals (Probably Just Space Noise)",
    "Marathon Runner Wins Race, Immediately Begins Training for Next One Like a Maniac",
    "Brewery Hopes Fancy Labels Distract You From the Fact That All Their Beer Tastes the Same",
    "Dentists Nationwide Suddenly Remember That Disabled People Exist",
    "Documentary Filmmaker Discovers Climate Change Is Bad—More at 11",
    "Online Learning Platform Expands Reach (Because No One Can Afford College Anymore)",
    "Abandoned Lots Turned Into Gardens—Until Developers Decide They're 'Prime Real Estate'",
    "Award-Winning Architect Designs Building That Will Be Obsolete in 10 Years",
]

for i in range(30):
    print("Given one original headline and two different sarcastic version, choose the one that is more sarcastic. You should directly answer [A] or [B] without any additional text.")
    print("Original headline: " + examples[i])
    if i % 2 == 0:
        print("Sarcastic headline A: " + answers_1[i])
        print("Sarcastic headline B: " + answers_2[i])
        print("---------------------------------------------")
        print("A is from finetune")
    else:
        print("Sarcastic headline A: " + answers_2[i])
        print("Sarcastic headline B: " + answers_1[i])
        print("---------------------------------------------")
        print("B is from finetune")
    
    print("#############################################")


[
    "b"
    "b"
    "b"
    "a"
    "a"
    "b"
    "b"
    "b"
    "b"
    "a"
    "a"
    "a"
    "b"
    
]