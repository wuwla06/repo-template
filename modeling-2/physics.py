"""
Модуль для физических расчетов электрического поля
"""

import numpy as np
from constants import *

def calculate_bounds():
    """Автоматически вычисляет границы области по расположению зарядов"""
    if not CHARGES:
        return -5, 5, -5, 5
    
    x_coords = [charge[0] for charge in CHARGES]
    y_coords = [charge[1] for charge in CHARGES]
    
    x_min = min(x_coords) - AUTO_BOUNDS_MARGIN
    x_max = max(x_coords) + AUTO_BOUNDS_MARGIN
    y_min = min(y_coords) - AUTO_BOUNDS_MARGIN
    y_max = max(y_coords) + AUTO_BOUNDS_MARGIN
    
    # Гарантируем минимальный размер области
    if x_max - x_min < 2 * AUTO_BOUNDS_MARGIN:
        x_center = (x_min + x_max) / 2
        x_min = x_center - AUTO_BOUNDS_MARGIN
        x_max = x_center + AUTO_BOUNDS_MARGIN
    
    if y_max - y_min < 2 * AUTO_BOUNDS_MARGIN:
        y_center = (y_min + y_max) / 2
        y_min = y_center - AUTO_BOUNDS_MARGIN
        y_max = y_center + AUTO_BOUNDS_MARGIN
    
    return x_min, x_max, y_min, y_max

def electric_field(x, y, charges=None):
    """Вычисление вектора напряженности в точке (x, y)"""
    if charges is None:
        charges = CHARGES
    
    Ex, Ey = 0.0, 0.0
    
    for xi, yi, qi in charges:
        dx = x - xi
        dy = y - yi
        r_squared = dx**2 + dy**2
        
        # Защита от деления на ноль
        if r_squared < 1e-12:
            return np.inf, np.inf if qi > 0 else -np.inf, -np.inf
            
        r_pow_3_2 = r_squared * np.sqrt(r_squared)
        factor = K * qi / r_pow_3_2
        
        Ex += factor * dx
        Ey += factor * dy
    
    return Ex, Ey

def calculate_field_on_grid(x_min=None, x_max=None, y_min=None, y_max=None, resolution=None):
    """Вычисление поля на всей сетке"""
    if x_min is None or x_max is None or y_min is None or y_max is None:
        x_min, x_max, y_min, y_max = calculate_bounds()
    
    if resolution is None:
        # Автоматическое вычисление разрешения сетки
        x_range = x_max - x_min
        y_range = y_max - y_min
        resolution = max(30, int(30 * max(x_range, y_range) / 10))
    
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Векторизованное вычисление для скорости
    Ex = np.zeros_like(X)
    Ey = np.zeros_like(Y)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            ex, ey = electric_field(X[i, j], Y[i, j])
            Ex[i, j] = ex
            Ey[i, j] = ey
    
    Emag = np.sqrt(Ex**2 + Ey**2)
    
    # Заменяем бесконечные значения на максимальные конечные
    finite_max = np.max(Emag[np.isfinite(Emag)])
    Emag = np.where(np.isfinite(Emag), Emag, finite_max * 10)
    
    return X, Y, Ex, Ey, Emag

def is_near_charge(x, y, threshold=0.1):
    """Проверяет, находится ли точка слишком близко к какому-либо заряду"""
    for xi, yi, qi in CHARGES:
        if np.sqrt((x - xi)**2 + (y - yi)**2) < threshold:
            return True
    return False

def trace_field_line(start_x, start_y, direction=1, max_steps=None):
    """Трассировка линии поля методом Рунге-Кутта 4-го порядка"""
    if max_steps is None:
        max_steps = LINE_LENGTH
    
    points = []
    x, y = start_x, start_y
    ds = LINE_STEP * direction
    
    for step in range(max_steps):
        # Проверка на близость к зарядам
        if is_near_charge(x, y, threshold=0.15):
            break
            
        points.append((x, y))
        
        # Вычисляем поле
        Ex, Ey = electric_field(x, y)
        E_norm = np.sqrt(Ex**2 + Ey**2)
        
        # Критерии остановки
        if E_norm < 1e-4:
            break
            
        # Нормализуем вектор направления
        nx, ny = Ex/E_norm, Ey/E_norm
        
        # Метод Рунге-Кутта 4-го порядка
        try:
            # k1
            k1x, k1y = nx, ny
            
            # k2
            Ex2, Ey2 = electric_field(x + 0.5*ds*k1x, y + 0.5*ds*k1y)
            norm2 = np.sqrt(Ex2**2 + Ey2**2)
            if norm2 < 1e-4:
                break
            k2x, k2y = Ex2/norm2, Ey2/norm2
            
            # k3
            Ex3, Ey3 = electric_field(x + 0.5*ds*k2x, y + 0.5*ds*k2y)
            norm3 = np.sqrt(Ex3**2 + Ey3**2)
            if norm3 < 1e-4:
                break
            k3x, k3y = Ex3/norm3, Ey3/norm3
            
            # k4
            Ex4, Ey4 = electric_field(x + ds*k3x, y + ds*k3y)
            norm4 = np.sqrt(Ex4**2 + Ey4**2)
            if norm4 < 1e-4:
                break
            k4x, k4y = Ex4/norm4, Ey4/norm4
            
            # Обновляем координаты
            x += (ds/6.0) * (k1x + 2*k2x + 2*k3x + k4x)
            y += (ds/6.0) * (k1y + 2*k2y + 2*k3y + k4y)
            
        except (ZeroDivisionError, ValueError):
            break
        
        # Проверка границ
        x_min, x_max, y_min, y_max = calculate_bounds()
        if x < x_min or x > x_max or y < y_min or y > y_max:
            break
    
    return np.array(points) if points else np.array([])

def trace_all_field_lines():
    """Трассировка всех линий поля от всех зарядов"""
    lines = []
    x_min, x_max, y_min, y_max = calculate_bounds()
    
    for xi, yi, qi in CHARGES:
        # Определяем количество линий в зависимости от величины заряда
        n_lines = LINES_PER_CHARGE
        
        # Радиус старта зависит от величины заряда
        base_radius = START_RADIUS_FACTOR * min(2.0, abs(qi))
        radius = max(0.2, base_radius)  # Минимальный радиус 0.2
        
        for angle in np.linspace(0, 2*np.pi, n_lines, endpoint=False):
            start_x = xi + radius * np.cos(angle)
            start_y = yi + radius * np.sin(angle)
            
            # Пропускаем точки, которые сразу вне границ
            if (start_x < x_min or start_x > x_max or 
                start_y < y_min or start_y > y_max):
                continue
            
            # Определяем направление в зависимости от знака заряда
            if qi > 0:
                line = trace_field_line(start_x, start_y, direction=1)
            else:
                line = trace_field_line(start_x, start_y, direction=-1)
            
            # Добавляем линию, если она достаточно длинная
            if len(line) > MIN_LINE_LENGTH:
                lines.append(line)
    
    return lines