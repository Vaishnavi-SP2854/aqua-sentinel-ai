from django.shortcuts import render

def index_view(request):
    contaminants = [
        {'icon':'💧','name':'Safe',        'desc':'Meets WHO standards'},
        {'icon':'🦠','name':'Bacterial',   'desc':'Typhoid, cholera risk'},
        {'icon':'⚗️','name':'Chemical',    'desc':'Industrial discharge'},
        {'icon':'🔩','name':'Heavy Metal', 'desc':'Lead, arsenic, mercury'},
        {'icon':'🚽','name':'Sewage',      'desc':'Fecal contamination'},
    ]
    return render(request, 'index.html', {'contaminants': contaminants})

def report_form_view(request): return render(request, 'report_form.html')
def result_view(request):      return render(request, 'result.html')
def map_view(request):         return render(request, 'map.html')
def dashboard_view(request):   return render(request, 'dashboard.html')
def login_view(request):       return render(request, 'login.html')
def register_view(request):    return render(request, 'register.html')