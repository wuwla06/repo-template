"""
Модуль для визуализации электрического поля
"""

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Circle
import numpy as np
from constants import CHARGES
from physics import calculate_bounds

def plot_charges(ax):
    """Рисует заряды на указанной оси с информацией"""
    for idx, (xi, yi, qi) in enumerate(CHARGES):
        # Цвет в зависимости от знака заряда
        color = 'red' if qi > 0 else 'blue'
        
        # Размер зависит от величины заряда
        radius = 0.15 * min(2.0, abs(qi))
        
        # Рисуем круг заряда
        circle = Circle((xi, yi), radius, color=color, 
                       ec='black', linewidth=2, zorder=10)
        ax.add_patch(circle)
        
        # Знак заряда внутри
        sign = '+' if qi > 0 else '−'
        ax.text(xi, yi, sign, ha='center', va='center',
                fontsize=12, fontweight='bold', color='white', zorder=11)
        
        # Подпись с номером и величиной заряда
        ax.text(xi, yi + radius*1.8, f'q{idx+1}={qi:+.1f}',
                ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", 
                         facecolor='lightyellow', alpha=0.8, edgecolor='gray'),
                zorder=11)

def create_field_map(ax, X, Y, Ex, Ey, Emag):
    """Создает карту поля со стрелками"""
    # Используем логарифмическую шкалу для лучшего отображения
    norm = colors.LogNorm(vmin=Emag[Emag > 0].min(), vmax=Emag.max())
    
    # Карта модуля поля
    im = ax.contourf(X, Y, Emag, levels=50, cmap='hot', norm=norm, alpha=0.7)
    
    # Векторные стрелки (только где поле достаточно сильное)
    mask = Emag > Emag.max() * 0.01
    step = max(1, X.shape[0] // 20)
    
    if mask.any():
        # Нормализуем стрелки для единообразного отображения
        Ex_norm = np.where(mask, Ex/Emag, 0)
        Ey_norm = np.where(mask, Ey/Emag, 0)
        
        ax.quiver(X[::step, ::step], Y[::step, ::step],
                 Ex_norm[::step, ::step], Ey_norm[::step, ::step],
                 color='white', alpha=0.6, scale=40, width=0.003,
                 headwidth=3, headlength=5)
    
    # Рисуем заряды
    plot_charges(ax)
    
    # Устанавливаем границы
    x_min, x_max, y_min, y_max = calculate_bounds()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    
    ax.set_title('Карта модуля поля и векторные стрелки')
    ax.set_xlabel('Координата X')
    ax.set_ylabel('Координата Y')
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    
    return im

def create_field_lines(ax, lines):
    """Создает график линий поля"""
    if not lines:
        ax.text(0.5, 0.5, 'Нет линий поля для отображения', 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    # Рисуем линии разным цветом в зависимости от заряда-источника
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(CHARGES))))
    
    for line in lines:
        # Цвет в зависимости от длины линии
        line_length = len(line)
        alpha = min(0.7, 0.3 + 0.4 * (line_length / 500))
        
        ax.plot(line[:, 0], line[:, 1], 'b-', 
                linewidth=1.0, alpha=alpha, zorder=5)
    
    # Рисуем заряды поверх линий
    plot_charges(ax)
    
    # Устанавливаем границы
    x_min, x_max, y_min, y_max = calculate_bounds()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    
    ax.set_title('Линии напряженности электрического поля')
    ax.set_xlabel('Координата X')
    ax.set_ylabel('Координата Y')
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    
    # Добавляем легенду с информацией о зарядах
    charge_info = []
    for idx, (xi, yi, qi) in enumerate(CHARGES):
        sign = '+' if qi > 0 else '−'
        charge_info.append(f'q{idx+1} ({sign}{abs(qi)})')
    
    ax.text(0.02, 0.98, f'Заряды: {", ".join(charge_info)}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", 
                     facecolor='white', alpha=0.8),
            verticalalignment='top')