from django.shortcuts import render

def home(request):
    return render(request, 'another_app_name/home.html', {})