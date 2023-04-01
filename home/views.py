from django.shortcuts import render, HttpResponse
import joblib
import numpy as np

def index(request):
    return render(request, 'index.html')


def result(request):
    cls = joblib.load('final_model.sav')
    experience = request.GET['experience']
    experience = np.reshape(experience, (-1, 1))
    experience = experience.astype('float64')
    # print(experience)
    result = cls.predict(experience)
    result = result.item()
    result = round(result, 2)
    # print(result)
    return render(request, 'result.html', {'result': result})