import csv
from django.shortcuts import render

def doctors_details(request):
    csv_file_path = "E:\Medchoicehub\MyProject\static\doctors.csv"  # Update with your CSV file path
    doctors = []

    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                doctors.append(row)
    except FileNotFoundError:
        print("CSV file not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return render(request, 'doctors.html', {'doctors': doctors})

