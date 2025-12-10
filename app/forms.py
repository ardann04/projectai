from django import forms

class WorkoutForm(forms.Form):
    Age = forms.IntegerField(label='Umur')
    Gender = forms.ChoiceField(choices=[('Male', 'Laki-laki'), ('Female', 'Perempuan')])
    Weight_kg = forms.FloatField(label='Berat Badan (kg)')
    Height_m = forms.FloatField(label='Tinggi Badan (m)')
    Session_Duration_hours = forms.FloatField(label='Durasi Sesi (jam)')
    Water_Intake_liters = forms.FloatField(label='Asupan Air (liter)')
    Workout_Frequency_days_per_week = forms.IntegerField(label='Frekuensi Workout (hari/minggu)')
    Experience_Level = forms.ChoiceField(choices=[('Beginner', 'Pemula'), ('Intermediate', 'Menengah'), ('Advanced', 'Lanjutan')])
