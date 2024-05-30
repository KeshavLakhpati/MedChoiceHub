import csv
from django.shortcuts import render

def lab_details(request):
    csv_file_path = "E:\Medchoicehub\MyProject\static\labs_data.csv"  # Update with your CSV file path
    labs_list = []

    with open(csv_file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            labs_list.append(row)

    return render(request, 'laboratory.html', {'laboratory': labs_list})

