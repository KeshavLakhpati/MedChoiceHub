# Importing libraries
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_PATH = "E:\Medchoicehub\MyProject\static\Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis=1)


# Encoding the target value into numerical
# value using LabelEncoder
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

#Splitting the data for training and testing the model
X = data.iloc[:,:-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test =train_test_split(
X, y, test_size = 0.2, random_state = 24)

# print(f"Train: {X_train.shape}, {y_train.shape}")
# print(f"Test: {X_test.shape}, {y_test.shape}")

# Training and testing Random Forest Classifier
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)
preds = rf_model.predict(X_test)
# print(f"Accuracy on train data by Random Forest Classifier\
# : {accuracy_score(y_train, rf_model.predict(X_train)) * 100}")

# print(f"Accuracy on test data by Random Forest Classifier\
# : {accuracy_score(y_test, preds) * 100}")


# Training the models on whole data
final_rf_model = RandomForestClassifier(random_state=18)
final_rf_model.fit(X, y)

# Reading the test data
test_data = pd.read_csv("E:\Medchoicehub\MyProject\static\Testing.csv").dropna(axis=1)
test_X = test_data.iloc[:, :-1]
test_Y = encoder.transform(test_data.iloc[:, -1])

# Making prediction by take mode of predictions
# made by all the classifiers
rf_preds = final_rf_model.predict(test_X)

final_preds = [mode(k)[0] for k in zip(rf_preds)]

# print(f"Accuracy on Test dataset by the combined model\
# : {accuracy_score(test_Y, final_preds) * 100}")


symptoms = X.columns.values

# Creating a symptom index dictionary to encode the
# input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index


data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}

# Defining the Function
# Input: string containing symptoms separated by commas
# Output: Generated predictions by models
def predictDisease(symptoms):
    symptoms = symptoms.split(",")

    # creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1

    # reshaping the input data and converting it
    # into suitable format for model predictions
    input_data = np.array(input_data).reshape(1, -1)

    # generating individual outputs
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]

    # making final prediction by taking mode of all predictions
    #final_prediction = mode(rf_prediction)[0][0]
    return rf_prediction


# Testing the function
#print(predictDisease("Itching,Skin Rash,Nodal Skin Eruptions"))
user_input = input("Enter symptoms separated by commas: ")
print("Your disease is: ",predictDisease(user_input))

#----------------------------------------------------------------------------------------------------------------------------------------------

# Load the doctors dataset
# doctors_data = pd.read_csv("E:\Medchoicehub\MyProject\static\doctors.csv")


# # df['Specialization'].unique()
# # Dermatology = df.loc[df['Specialization'] == 'Dermatology', :]
# # dermatogy_json = Dermatology.to_json(orient='records')

# # Initialize a dictionary to map dise
# # ases with specializations
# disease_specialization_mapping = {
#     'Fungal infection': 'Dermatologist',
#     'Allergy': 'Allergist',
#     'GERD': 'Gastroenterologist',
#     'Chronic cholestasis': 'Hepatologist',
#     'Drug Reaction': 'Dermatologist',
#     'Peptic ulcer disease': 'Gastroenterology',
#     'AIDS': 'Infectious Disease Specialist',
#     'Diabetes': 'Endocrinologist',
#     'Gastroenteritis': 'Gastroenterologist',
#     'Bronchial Asthma': 'Pulmonologist',
#     'Hypertension': 'Cardiologist',
#     'Migraine': 'Neurologist',
#     'Cervical spondylosis': 'Neurologist',
#     'Paralysis (brain hemorrhage)': 'Neurologist',
#     'Jaundice': 'Gastroenterologist',
#     'Malaria': 'Infectious Disease Specialist',
#     'Chicken pox': 'Infectious Disease Specialist',
#     'Dengue': 'Infectious Disease Specialist',
#     'Typhoid': 'Infectious Disease Specialist',
#     'hepatitis A': 'Gastroenterologist',
#     'Hepatitis B': 'Gastroenterologist',
#     'Hepatitis C': 'Gastroenterologist',
#     'Hepatitis D': 'Gastroenterologist',
#     'Hepatitis E': 'Gastroenterologist',
#     'Alcoholic hepatitis': 'Gastroenterologist',
#     'Tuberculosis': 'Gastroenterologist',
#     'Common Cold': 'ENT/general physician',
#     'Pneumonia': 'Pulmonologist',
#     'Dimorphic hemmorhoids(piles)': 'Proctologist',
#     'Heart attack': 'Cardiologist',
#     'Varicose veins': 'Phlebologist',
#     'Hypothyroidism': 'Endocrinologist',
#     'Hyperthyroidism': 'Endocrinologist',
#     'Hypoglycemia': 'Endocrinologist',
#     'Osteoarthristis': 'Rheumatologist',
#     'Arthritis': 'Rheumatologist',
#     'Vertigo Paroxysmal Positional Vertigo': 'Otolaryngologist',
#     'Acne': 'Dermatologist',
#     'Urinary tract infection': 'Urologist',
#     'Psoriasis': 'Dermatologist',
#     'Impetigo': 'Dermatologist'
# }

# # doctors_data = pd.read_csv("doctors.csv")
# # doc_detals = list(doctors_data)

# # Initialize a dictionary to map specializations with doctor details
# specialization_doctor_mapping = {
#     'Dermatologist': {
#         'ID': [16, 38, 121, 126, 130, 186, 218, 255, 356, 388,407, 414, 463, 508, 547, 643, 696, 697, 780, 828,861, 881, 897, 921, 994, 997],
#         'Name': ["Avani Rajput", "Abhinav Chauhan", "Abhinav Chauhan", "Aditi Das", "Eshaan Yadav","Daksh Tiwari", "Bhavya Shah", "Aaradhya Bansal", "Arjun Patel", "Avani Rajput","Aaradhya Bansal", "Arnav Khobe", "Samyak Nagdevte", "Advik Gupta", "Ananya Mishra","Ananya Mishra", "Abhinav Chauhan", "Aditi Das", "Aditi Das", "Avani Rajput", "Aditi Das","Abhinav Chauhan", "Arjun Patel", "Ananya Mishra", "Abhinav Chauhan", "Bhavya Shah"],
#         'Qualification': ["DC", "EdD", "PhD", "DPT", "DNP", "PharmD", "JD", "PhD", "PhD","MBBS", "JD", "MD-PhD", "DPT", "MD-PhD", "PharmD", "DO", "JD","PhD", "MBBS", "DC", "MBBS", "DNP", "DNP", "DVM", "DVM", "DMD"],
#         'Specialization':['Dermatologist'] * 27,
#         'Contact':["9876543216", "9876543238", "98765432121", "98765432126", "98765432130","98765432186", "98765432218", "98765432255", "98765432356", "98765432388", "98765432407", "98765432414", "98765432463", "98765432508", "98765432547","98765432643", "98765432696", "98765432697", "98765432780", "98765432828","98765432861", "98765432881", "98765432897", "98765432921", "98765432994","98765432997"],
#         'Locality': ["Moshi", "Nigdi", "Kharadi", "Hinjewadi", "Wakad Chowk","Bavdhan", "Firodiya Nagar", "Bund Garden", "Viman Nagar","Theur", "Ganeshwadi", "Tingale Nagar", "Pimple Nilach", "Tingale Nagar", "Hadapsar", "Dhankawadi", "Kalyani Nagar","Viman Nagar", "Wakad", "Ravet", "Wakad", "Model Colony","Wakad", "Kharadi", "Deccan Gymkhana", "Warje"],
#         'Address': ["Pune, Maharashtra"] * 27
#     },
#     'Gastroenterologist': {
#         'ID': [20, 30, 41, 56, 188, 212, 318, 330, 334, 404, 437, 527, 551, 674, 680, 713, 740, 853, 871, 914, 962, 993],
#         'Name': ['Bhavya Shah', 'Aditya Nardaswar', 'Pranay Maheta', 'Daksh Tiwari', 'Deepika Verma', 'Aditi Das', 'Deepika Verma', 'Abhinav Chauhan', 'Avani Rajput', 'Arjun Patel', 'Akshay Kumar', 'Advik Gupta', 'Arjun Patel', 'Abhinav Chauhan', 'Manthan Pal', 'Arjun Patel', 'Aaradhya Bansal', 'Abhinav Chauhan', 'Aarav Agarwal', 'Avani Rajput', 'Aditi Das', 'Avani Rajput'],
#         'Qualification': ['PharmD', 'MD-PhD', 'PharmD', 'DVM', 'DO', 'DMD', 'PharmD', 'MD', 'MBBS', 'MBBS', 'JD', 'DPT', 'DNP', 'DVM', 'DPT', 'DC', 'DC', 'DVM', 'BDS', 'PsyD', 'DMD', 'PsyD'],
#         'Specilization':["Gastroenterologist"]*22,
#         'Contact': ['9876543220', '9876543230', '9876543241', '9876543256', '98765432188', '98765432212', '98765432318', '98765432330', '98765432334', '98765432404', '98765432437', '98765432527', '98765432551', '98765432674', '98765432680', '98765432713', '98765432740', '98765432853', '98765432871', '98765432914', '98765432962', '98765432993'],
#         'Locality': ['Nigdi', 'Sahakar Nagar', 'Swargate', 'Theur', 'NIBM (Nanded City)', 'Kondhwa Khurd', 'Moshi', 'Dhayari', 'Ganeshwadi', 'Bibwewadi', 'Kharadi', 'Fatima Nagar', 'Firodiya Nagar', 'NIBM (Nanded City)', 'Model Colony', 'Mundhwa', 'Shivaji Nagar', 'Koregaon Park', 'Hadapsar', 'Mohammadwadi', 'Mundhwa', 'Aundh Road'],
#         'Address': ["Pune, Maharashtra"] * 27
#     },
#     'Endocrinologist': {
#         'ID': [1, 45, 66, 94, 122, 124, 181, 210, 235, 242, 275, 440, 590, 591, 648, 652, 662, 714, 741, 766, 811, 836, 863, 879, 886, 889, 948, 984],
#         'Name': ["Avani Rajput", "Eshaan Yadav", "Advik Gupta", "Advik Gupta", "Chetan Singh", "Arjun Patel", "Pranay Maheta", "Akshay Kumar", "Akshay Kumar", "Bhavya Shah", "Ananya Mishra", "Bhavya Shah", "Arjun Patel", "Eshaan Yadav","Ananya Mishra", "Aisha Joshi", "Ananya Mishra", "Avani Rajput", "Deepika Verma", "Aaradhya Bansal","Akshay Kumar", "Advik Gupta", "Aditi Das", "Eshaan Yadav", "Ananya Mishra", "Chetan Singh", "Eshaan Yadav","Akshay Kumar"],
#         'Qualification': ["JD", "PharmD", "PsyD", "MD-PhD", "DO", "DC", "BDS", "DNP", "DO", "BDS", "BDS", "MD", "MBBS", "MD-PhD", "PsyD", "MD-PhD", "PharmD", "PharmD", "BDS", "DO", "PsyD", "DO", "MBBS", "MBBS", "MD-PhD", "DMD", "DVM", "JD"],
#         'Specialiaztion':['Endocrinologist']*28,
#         'Contact': ["987654321", "9876543245", "9876543266", "9876543294", "98765432122", "98765432124", "98765432181","98765432210", "98765432235", "98765432242", "98765432275", "98765432440", "98765432590", "98765432591","98765432648", "98765432652", "98765432662", "98765432714", "98765432741", "98765432766", "98765432811","98765432836", "98765432863", "98765432879", "98765432886", "98765432889", "98765432948", "98765432984"],
#         'Locality': ["Kalyani Nagar", "Dhankawadi", "Kharadi", "Dhankawadi", "Akurdi", "Fatima Nagar", "Pimple Gaurav", "Dhanori",                   "Moshi", "Kothrud", "Aundh Road", "Shivaji Nagar", "Fatima Nagar", "Kalyani Nagar", "Firodiya Nagar", "Aundh","Kondhwa Khurd", "Kharadi", "Dhankawdi", "Ravet", "Theur", "Magarpatta", "Mohammadwadi", "Bavdhan", "Warje",                    "Pimple Nilach", "Kalyani Nagar", "Akurdi"],
#         'Address': ["Pune, Maharashtra"] * 28
#     },
#     'Pulmonologist': {
#         'ID': [817, 823, 826, 835, 840, 846, 893, 929, 933],
#         'Name': ["Aaradhya Bansal", "Manthan Pal", "Avani Rajput", "Deepika Verma","Aisha Joshi", "Aditi Das", "Abhinav Chauhan", "Akshay Kumar", "Aditi Das"],
#         'Qualification': ["DO", "DVM", "JD", "BDS", "PharmD", "DPT", "DC", "DO", "PsyD"],
#         'Specialiaztion':['Pulmonologist']*9,
#         'Contact': ["98765432817", "98765432823", "98765432826", "98765432835", "98765432840","98765432846", "98765432893", "98765432929", "98765432933"],
#         'Locality': ["Kondhwa", "Warje", "Warje", "Pimple Nilakh", "Hadapsar", "Firodiya Nagar","Akurdi", "Wakad", "Lohegaon"],
#         'Address': ["Pune, Maharashtra"] * 9
#     },
#     'Cardiologist': {
#         'ID': [61, 139, 211, 240, 269, 285, 361, 382, 413, 453, 459, 472, 482, 499, 516, 528, 544, 570, 715, 767, 784, 851, 873, 998],
#         'Name': ['Chetan Singh', 'Aisha Joshi', 'Abhinav Chauhan', 'Eshaan Yadav', 'Aaradhya Bansal', 'Bhavya Shah', 'Bhavya Shah', 'Daksh Tiwari', 'Arjun Patel', 'Chetan Singh', 'Advik Gupta', 'Abhinav Chauhan', 'Aaradhya Bansal', 'Akshay Kumar', 'Deepika Verma', 'Bhavya Shah', 'Daksh Tiwari', 'Aaradhya Bansal', 'Arjun Patel', 'Advik Gupta', 'Advik Gupta', 'Avani Rajput', 'Aarav Agarwal', 'Arjun Patel'],
#         'Qualification': ['PhD', 'DNP', 'MD', 'MBBS', 'PharmD', 'PharmD', 'PsyD', 'MD-PhD', 'DO', 'BDS', 'DMD', 'DMD', 'MD-PhD', 'MBBS', 'MBBS', 'PsyD', 'DVM', 'DPT', 'DC', 'MD-PhD', 'MD-PhD', 'JD', 'JD', 'EdD'],
#         'Specialiaztion':['Cardiologist']*24,
#         'Contact': ['9876543261', '98765432139', '98765432211', '98765432240', '98765432269', '98765432285', '98765432361', '98765432382', '98765432413', '98765432453', '98765432459', '98765432472', '98765432482', '98765432499', '98765432516', '98765432528', '98765432544', '98765432570', '98765432715', '98765432767', '98765432784', '98765432851', '98765432873', '98765432998'],
#         'Locality': ['Bhosari', 'Camp', 'Bibwewadi', 'Balewadi', 'Dhayari', 'Wakad', 'Bavdhan', 'Lohegaon', 'Yerwada', 'Pune Satara Road', 'Bavdhan', 'Pimple Nilach', 'Theur', 'Baner', 'Aundh Road', 'Camp', 'Kondhwa Khurd', 'Pune Satara Road', 'Katraj', 'Dhankawdi', 'Kalyani Nagar', 'Model Colony', 'Shivaji Nagar', 'Aundh Road'],
#         'Address': ["Pune, Maharashtra"] * 24
#     },
#     'Neurologist': {
#         'ID': [75, 90, 114, 219, 252, 256, 556, 567, 584, 588, 600, 624, 743, 775, 778, 810, 859, 887, 901, 981, 999],
#         'Name': ['Chetan Singh', 'Bhavya Shah', 'Aaradhya Bansal', 'Abhinav Chauhan', 'Aditi Das', 'Aditi Das', 'Bhavya Shah', 'Deepika Verma', 'Advik Gupta', 'Ananya Mishra', 'Daksh Tiwari', 'Eshaan Yadav', 'Deepika Verma', 'Avani Rajput', 'Deepika Verma', 'Akshay Kumar', 'Bhavya Shah', 'Aditi Das', 'Aditi Das', 'Eshaan Yadav', 'Daksh Tiwari'],
#         'Qualification': ['MBBS', 'BDS', 'MD', 'BDS', 'PsyD', 'BDS', 'PsyD', 'DPT', 'MD-PhD', 'DNP', 'DC', 'DC', 'DPT', 'BDS', 'BDS', 'JD', 'MD-PhD', 'DC', 'EdD', 'DVM', 'MBBS'],
#         'Specilization':['Neurologist']*21,
#         'Contact': ['9876543275', '9876543290', '98765432114', '98765432219', '98765432252', '98765432256', '98765432556', '98765432567', '98765432584', '98765432588', '98765432600', '98765432624', '98765432743', '98765432775', '98765432778', '98765432810', '98765432859', '98765432887', '98765432901', '98765432981', '98765432999'],
#         'Locality': ['Mundhwa', 'Firodiya Nagar', 'Ganeshwadi', 'Fatima Nagar', 'Wakad', 'Kondhwa', 'Ravet', 'Hinjewadi', 'Koregaon Park', 'Dhankawadi', 'Hadapsar', 'Pimpri-Chinchwad', 'Hinjewadi', 'Viman Nagar', 'Dhankawadi', 'Yerwada', 'Mundhwa', 'Shivaji Nagar', 'Camp', 'Baner', 'Dhankawadi'],
#         'Address': ["Pune, Maharashtra"] * 21
#     },
#     'ENT/general physician': {
#         'ID': [31, 91, 132, 133, 214, 225, 243, 249, 323, 341, 389, 457, 497, 519, 543, 554, 569, 619, 646, 669, 720, 755, 782, 795, 812, 842, 909, 918, 922, 945, 978, 982, 988],
#         'Name': ['Arjun Patel', 'Deepika Verma', 'Abhinav Chauhan', 'Pankaj Kumar', 'Aditi Das', 'Bhavya Shah', 'Ananya Mishra', 'Ananya Mishra', 'Deepika Verma', 'Aarav Agarwal', 'Aditi Das', 'Arjun Patel', 'Deepika Verma', 'Eshaan Yadav', 'Abhinav Chauhan', 'Arjun Patel', 'Avani Rajput', 'Daksh Tiwari', 'Aisha Joshi', 'Akshay Kumar', 'Arjun Patel', 'Aarav Agarwal', 'Aaradhya Bansal', 'Eshaan Yadav', 'Aarav Agarwal', 'Arjun Patel', 'Advik Gupta', 'Avani Rajput', 'Daksh Tiwari', 'Advik Gupta', 'Eshaan Yadav', 'Ananya Mishra', 'Avani Rajput'],
#         'Qualification': ['PhD', 'DMD', 'DPT', 'MBBS', 'DPT', 'JD', 'MBBS', 'PhD', 'MBBS', 'PsyD', 'DPT', 'DVM', 'MD', 'DNP', 'DNP', 'PharmD', 'PhD', 'DVM', 'PhD', 'MD-PhD', 'MBBS', 'DC', 'DPT', 'PharmD', 'DVM', 'PharmD', 'EdD', 'MD', 'PharmD', 'MD-PhD', 'MD-PhD', 'MBBS', 'BDS'],
#         'Specilization':['ENT/general physician']*33,
#         'Contact': ['9876543231', '9876543291', '98765432132', '98765432133', '98765432214', '98765432225', '98765432243', '98765432249', '98765432323', '98765432341', '98765432389', '98765432457', '98765432497', '98765432519', '98765432543', '98765432554', '98765432569', '98765432619', '98765432646', '98765432669', '98765432720', '98765432755', '98765432782', '98765432795', '98765432812', '98765432842', '98765432909', '98765432918', '98765432922', '98765432945', '98765432978', '98765432982', '98765432988'],
#         'Locality': ['Camp', 'Kondhwa', 'Kondhwa', 'Bhosari', 'Camp', 'Moshi', 'Dhankawdi', 'Pimple Gaurav', 'Baner', 'Kondhwa Khurd', 'Magarpatta', 'Kondhwa', 'Deccan Gymkhana', 'Koregaon Park', 'Hadapsar', 'Deccan Gymkhana', 'Moshi', 'Kalyani Nagar', 'Ravet', 'Fatima Nagar', 'Koregaon Park', 'Katraj', 'Ganeshwadi', 'Viman Nagar', 'Pimple Nilach', 'Shivaji Nagar', 'Kothrud', 'Dhayari', 'Model Colony', 'Baner', 'Pimpri-Chinchwad', 'Kalyani Nagar', 'NIBM (Nanded City)'],
#         'Address': ["Pune, Maharashtra"] * 33
#     },
#     'Proctologist': {
#         'ID':[36, 69, 148, 173, 175, 201, 280, 287, 297, 306, 317, 490, 549, 577, 587, 699, 711, 728, 735, 770],
#         'Name': ['Aarav Agarwal', 'Avani Rajput', 'Aaradhya Bansal', 'Aarav Agarwal', 'Akshay Kumar', 'Eshaan Yadav', 'Aarav Agarwal', 'Aarav Agarwal', 'Aarav Agarwal', 'Deepika Verma', 'Eshaan Yadav', 'Daksh Tiwari', 'Akshay Kumar', 'Arjun Patel', 'Daksh Tiwari', 'Aarav Agarwal', 'Eshaan Yadav', 'Eshaan Yadav', 'Advik Gupta', 'Arjun Patel'],
#         'Qualification': ['DPT', 'DC', 'MBBS', 'DVM', 'DC', 'MD', 'DO', 'DC', 'MD-PhD', 'PsyD', 'DO', 'JD', 'PhD', 'DNP', 'EdD', 'DC', 'PharmD', 'PsyD', 'DPT', 'MD-PhD'],
#         'Specilization':['Proctologist']*20,
#         'Contact': ['9876543236', '9876543269', '98765432148', '98765432173', '98765432175', '98765432201', '98765432280', '98765432287', '98765432297', '98765432306', '98765432317', '98765432490', '98765432549', '98765432577', '98765432587', '98765432699', '98765432711', '98765432728', '98765432735', '98765432770'],
#         'Locality': ['Model Colony', 'Warje', 'Shivaji Nagar', 'Bavdhan', 'Viman Nagar', 'Yerwada', 'Sahakar Nagar', 'Hadapsar', 'Kalyani Nagar', 'NIBM (Nanded City)', 'Mundhwa', 'Baner', 'Hinjewadi', 'Lohegaon', 'Kothrud', 'Magarpatta', 'Yerwada', 'Bibwewadi', 'Model Colony', 'Bibwewadi'],
#         'Address': ["Pune, Maharashtra"] * 20
#     },
#     'Phlebologist': {
#         'ID': [35, 71, 72, 76, 89, 95, 154, 223, 343, 378, 399, 428, 622, 629, 672, 736, 785, 805, 843, 858, 862, 957, 973],
#         'Name': ['Aisha Joshi', 'Advik Gupta', 'Bhavya Shah', 'Chetan Singh', 'Bhavya Shah', 'Bhavya Shah', 'Avani Rajput', 'Akshay Kumar', 'Daksh Tiwari', 'Aisha Joshi', 'Arjun Patel', 'Arjun Patel', 'Akshay Kumar', 'Deepika Verma', 'Chetan Singh', 'Avani Rajput', 'Arjun Patel', 'Abhinav Chauhan', 'Avani Rajput', 'Aarav Agarwal', 'Aditi Das', 'Eshaan Yadav', 'Deepika Verma'],
#         'Qualification': ['MD', 'PsyD', 'MD', 'DC', 'DNP', 'DPT', 'DPT', 'PsyD', 'MD', 'BDS', 'DVM', 'PsyD', 'MD', 'DC', 'DC', 'DO', 'MD', 'DVM', 'DC', 'MD-PhD', 'DC', 'MD', 'DC'],
#         'Specilization':['Phlebologist']*23,
#         'Contact': ['9876543235', '9876543271', '9876543272', '9876543276', '9876543289', '9876543295', '98765432154', '98765432223', '98765432343', '98765432378', '98765432399', '98765432428', '98765432622', '98765432629', '98765432672', '98765432736', '98765432785', '98765432805', '98765432843', '98765432858', '98765432862', '98765432957', '98765432973'],
#         'Locality': ['Viman Nagar', 'Kondhwa', 'Viman Nagar', 'Pimple Nilach', 'Kothrud', 'Wakad Chowk', 'Shivaji Nagar', 'Mundhwa', 'Ganeshwadi', 'Akurdi', 'Balewadi', 'Dhankawdi', 'Tingale Nagar', 'Hadapsar', 'Pimple Nilach', 'Camp', 'Balewadi', 'Bund Garden', 'Shivaji Nagar', 'Kothrud', 'Bavdhan', 'Camp', 'Balewadi'],
#         'Address': ["Pune, Maharashtra"] * 23
#     },
#     'Rheumatologist': {
#         'ID': [26, 340, 365, 393, 420, 422, 486, 511, 515, 671, 676, 683, 834, 867, 958],
#         'Name': ['Arjun Patel', 'Advik Gupta', 'Ananya Mishra', 'Aisha Joshi', 'Advik Gupta', 'Aisha Joshi', 'Bhavya Shah', 'Avani Rajput', 'Arjun Patel', 'Daksh Tiwari', 'Deepika Verma', 'Akshay Kumar', 'Chetan Singh', 'Bhavya Shah', 'Ananya Mishra'],
#         'Qualification': ['DVM', 'PharmD', 'DNP', 'MD-PhD', 'DMD', 'PhD', 'PhD', 'DNP', 'MD-PhD', 'JD', 'PsyD', 'JD', 'DC', 'DO', 'DC'],
#         'Specilization':['Rheumatologist']*15,
#         'Contact': ['9876543226', '98765432340', '98765432365', '98765432393', '98765432420', '98765432422', '98765432486', '98765432511', '98765432515', '98765432671', '98765432676', '98765432683', '98765432834', '98765432867', '98765432958'],
#         'Locality': ['Camp', 'Dhanori', 'Pimple Gaurav', 'Tingale Nagar', 'Theur', 'Bibwewadi', 'Bibwewadi', 'Koregaon Park', 'Shivaji Nagar', 'Bavdhan', 'Deccan Gymkhana', 'Sahakar Nagar', 'Ganeshwadi', 'Yerwada', 'Hadapsar'],
#         'Address': ["Pune, Maharashtra"] * 15
#     },
#     'Otolaryngologist': {
#         'ID': [7, 40, 83, 93, 103, 112, 131, 263, 322, 348, 355, 387, 403, 488, 545, 572, 595, 621, 681, 753, 820, 878, 950, 953, 1000],
#         'Name': ['Ananya Mishra', 'Ananya Mishra', 'Aditi Das', 'Aditi Das', 'Deepika Verma', 'Bhavya Shah', 'Advik Gupta', 'Aisha Joshi', 'Ananya Mishra', 'Aditi Das', 'Akshay Kumar', 'Advik Gupta', 'Aaradhya Bansal', 'Aaradhya Bansal', 'Arjun Patel', 'Aarav Agarwal', 'Ananya Mishra', 'Arjun Patel', 'Akshay Kumar', 'Akshay Kumar', 'Abhinav Chauhan', 'Advik Gupta', 'Daksh Tiwari', 'Advik Gupta', 'Bhavya Shah'],
#         'Qualification': ['DO', 'EdD', 'DC', 'DVM', 'DMD', 'DPT', 'DPT', 'PhD', 'MD', 'JD', 'DVM', 'JD', 'MD-PhD', 'BDS', 'MD', 'BDS', 'MBBS', 'PhD', 'DNP', 'DO', 'MD', 'DC', 'JD', 'PharmD', 'DNP'],
#         'Specilization':['Otolaryngologist']*25,
#         'Contact': ['987654327', '9876543240', '9876543283', '9876543293', '98765432103', '98765432112', '98765432131', '98765432263', '98765432322', '98765432348', '98765432355', '98765432387', '98765432403', '98765432488', '98765432545', '98765432572', '98765432595', '98765432621', '98765432681', '98765432753', '98765432820', '98765432878', '98765432950', '98765432953', '988000000000'],
#         'Locality': ['Hadapsar', 'Kharadi', 'Pimpri-Chinchwad', 'Hinjewadi', 'Viman Nagar', 'Ravet', 'Akurdi', 'Mundhwa', 'Model Colony', 'Shivaji Nagar', 'Ganeshwadi', 'Dhanori', 'Warje', 'Model Colony', 'Hinjewadi', 'Warje', 'Bavdhan', 'Wakad', 'Aundh Road', 'Swargate', 'Nigdi', 'NIBM (Nanded City)', 'Pimple Gaurav', 'Akurdi', 'Katraj'],
#         'Address': ["Pune, Maharashtra"] * 25
#     }
#     # 'ENT/general physician': {
#     #     'ID':[],
#     #     'Name': [],
#     #     'Qualification': [],
#     #     'Specilization':['ENT/general physician']*33,
#     #     'Contact':[],
#     #     'Locality': [],
#     #     'Address': ["Pune, Maharashtra"] * 33
#     # }
    
#     # Add more specialization-doctor mappings here
# }



# # Function to recommend a doctor based on the predicted disease
# def recommend_doctor(predicted_disease):
#     # Get the specialization for the predicted disease
#     specialization = disease_specialization_mapping.get(predicted_disease)
    
#     if specialization is not None:
#         # Get doctor details based on the specialization
#         doctor_details = specialization_doctor_mapping.get(specialization)
#         return doctor_details
#     else:
#         # If specialization mapping not found, return None
#         return None

# # # Example usage
# # predictDisease = predictDisease()  # Replace 'Disease_A' with the actual predicted disease
# predictDisease = input("Enter the predicted disease: ")
# # Recommend a doctor based on the predicted disease using the dictionary mapping
# recommended_doctor_details = recommend_doctor(predictDisease)


# if recommended_doctor_details is not None:
#     print('Recommended Doctor Details:')
#     print(f'ID: {recommended_doctor_details["ID"]}')
#     print(f'Name: {recommended_doctor_details["Name"]}')
#     #print(f'Specialization: {recommended_doctor_details["Specialization"]}')  
#     print(f'Qualification: {recommended_doctor_details["Qualification"]}')
#     print(f'Locality: {recommended_doctor_details["Locality"]}')
#     print(f'Address: {recommended_doctor_details["Address"]}')
# else:
#     print('No doctor available for the predicted disease')

# # Extract data from the dictionary
# # if specialization_doctor_mapping == 'Dermatologist':
# #     doctor_data = specialization_doctor_mapping['Dermatologist']
# #     ids = doctor_data['ID']
# #     names = doctor_data['Name']
# #     qualifications = doctor_data['Qualification']
# #     specializations = doctor_data['Specialization']
# #     contacts = doctor_data['Contact']
# #     addresss = doctor_data['Address']
# #     localitys = doctor_data['Locality']

# #     # Check that the lengths of doctor details lists are the same
# #     if len(names) == len(qualifications):
# #         for idx in range(len(names)):
# #             id = ids[idx]
# #             name = names[idx]
# #             qualification = qualifications[idx]
# #             specialization = specializations[idx]
# #             contact = contacts[idx]
# #             address = addresss[idx]
# #             locality = localitys[idx]
# #             print('Hello2')
# #             print(f"ID: {id}, Name: {name}, Qualification: {qualification}, Contact: {contact}, Qualification: {qualification}, Specialization: {specialization} Contact: {contact}, Address: {address}, Locality: {locality}")
# #     else:
# #         print("No doctors are available")

# selected_specialization = predictDisease

# if selected_specialization in specialization_doctor_mapping:
#     doctor_data = specialization_doctor_mapping[selected_specialization]
#     ids = doctor_data['ID']
#     names = doctor_data['Name']
#     qualifications = doctor_data['Qualification']
#     #specializations = doctor_data['Specialization']
#     contacts = doctor_data['Contact']
#     addresses = doctor_data['Address']
#     localities = doctor_data['Locality']

#     if len(names) == len(qualifications):
#         print(f"Specialization: {selected_specialization}")
#         for idx in range(len(names)):
#             id = ids[idx]
#             name = names[idx]
#             qualification = qualifications[idx]
#             # specialization = specializations[idx]
#             contact = contacts[idx]
#             address = addresses[idx]
#             locality = localities[idx]
#             print(f"ID: {id}, Name: {name}, Qualification: {qualification}, Contact: {contact}, Address: {address}, Locality: {locality}")    
#     else:
#         print(f"No doctors are available for {selected_specialization}")
# # else:
# #     print(f"Selected specialization '{selected_specialization}' not found")