from django.db import models

# Create your models here.
class Register(models.Model):
    username = models.CharField("username", max_length=20)
    password = models.CharField('password', max_length=20)



class value(models.Model):
    date_time = models.DateTimeField()
    supply_air_temp =  models.FloatField()
    heating_setpoint = models.FloatField()
    cooling_setpoint = models.FloatField()
    outdoor_air_temp = models.FloatField()
    mixed_air_temp =   models.FloatField()
    return_air_temp =  models.FloatField()
    supply_air_fan_status = models.FloatField()
    supply_air_fan_speed =  models.FloatField()
    outdoor_air_damper =    models.FloatField()
    return_air_damper =     models.FloatField()
    exhaust_air_damper = models.FloatField()
    cooling_coil_valve = models.FloatField()
    heating_coil_valve = models.FloatField()
    occupancy_mode =  models.FloatField()
    fault_detection = models.FloatField()