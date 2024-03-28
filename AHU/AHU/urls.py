from django.contrib import admin
from django.urls import path
from AHU import view
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',view.index),
    path('home/',view.index,name='home'),
    path('login/',view.login,name='login'),
    path('logout/',view.logout,name='logout'),
    path('dashboard/',view.dash,name='dashboard'),
    path('dashboardt/<int:sec>/',view.dash,name='dashboardt'),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)