from django.contrib import admin
from work.models import Register


class RegisterAdmin(admin.ModelAdmin):
    list_display = ('username', 'password')

admin.site.register(Register, RegisterAdmin)
# Register your models here.
