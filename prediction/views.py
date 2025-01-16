from django.shortcuts import redirect, render

# Create your views here.
# def nasdaq(request):
#     return render(request, 'prediction/nasdaq.html')

def main_redirect(request):
    return redirect("http://localhost:8501")