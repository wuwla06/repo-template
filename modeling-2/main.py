import os
import sys
import matplotlib.pyplot as plt
from constants import *
from physics import calculate_bounds, calculate_field_on_grid, trace_all_field_lines
from visualization import create_field_map, create_field_lines

def print_header():
    print("=" * 70)
    print("МОДЕЛИРОВАНИЕ ЭЛЕКТРИЧЕСКОГО ПОЛЯ ТОЧЕЧНЫХ ЗАРЯДОВ")
    print("Курс ФИТиП 2025")
    print("=" * 70)

def print_charge_info():
    print("\nКОНФИГУРАЦИЯ ЗАРЯДОВ:")
    print("-" * 40)
    for i, (x, y, q) in enumerate(CHARGES, 1):
        sign = "+" if q > 0 else "-"
        print(f"Заряд {i}: ({x:.1f}, {y:.1f}), величина: {sign}{abs(q):.1f}")
    print("-" * 40)

def print_summary(lines, result_path):
    print("\n" + "=" * 70)
    print("МОДЕЛИРОВАНИЕ ЗАВЕРШЕНО!")
    print("=" * 70)
    print(f"Количество зарядов: {len(CHARGES)}")
    print(f"Количество линий поля: {len(lines)}")
    
    x_min, x_max, y_min, y_max = calculate_bounds()
    print(f"Область моделирования: X[{x_min:.1f}, {x_max:.1f}], Y[{y_min:.1f}, {y_max:.1f}]")
    
    total_lines = sum(len(line) for line in lines)
    print(f"Всего точек в линиях: {total_lines}")
    print(f"Результат сохранен в файле: {result_path}")
    print("=" * 70)

def ensure_results_dir():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Создана папка для результатов: {RESULTS_DIR}")
    return os.path.join(RESULTS_DIR, RESULT_FILENAME)

def validate_charges():
    if not CHARGES:
        print("ОШИБКА: Не заданы заряды!")
        sys.exit(1)
    
    for i, charge in enumerate(CHARGES, 1):
        if len(charge) != 3:
            print(f"ОШИБКА: Заряд {i} имеет неправильный формат. Ожидается (x, y, q)")
            sys.exit(1)
        
        x, y, q = charge
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(q, (int, float))):
            print(f"ОШИБКА: Заряд {i} содержит нечисловые значения")
            sys.exit(1)

def main():
    print_header()
    
    # 0. Проверка корректности данных
    validate_charges()
    print_charge_info()
    
    # 1. Подготовка папки для результатов
    result_path = ensure_results_dir()
    
    # 2. Автоматическое вычисление границ
    x_min, x_max, y_min, y_max = calculate_bounds()
    print(f"\n1. Область моделирования:")
    print(f"   X: [{x_min:.2f}, {x_max:.2f}]")
    print(f"   Y: [{y_min:.2f}, {y_max:.2f}]")
    
    # 3. Вычисление поля на сетке
    print("\n2. Вычисление поля на сетке...")
    X, Y, Ex, Ey, Emag = calculate_field_on_grid(x_min, x_max, y_min, y_max)
    print(f"   Размер сетки: {X.shape[0]}×{X.shape[1]}")
    print(f"   Мин/макс напряженность: {Emag.min():.2e}/{Emag.max():.2e}")
    
    # 4. Трассировка линий поля
    print("\n3. Трассировка линий поля...")
    lines = trace_all_field_lines()
    print(f"   Сгенерировано линий: {len(lines)}")
    
    # 5. Создание визуализации
    print("\n4. Создание визуализации...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = axes
    
    # Левый график: карта модуля поля со стрелками
    im = create_field_map(ax1, X, Y, Ex, Ey, Emag)
    plt.colorbar(im, ax=ax1, label='Модуль напряженности |E| (лог. шкала)')
    
    # Правый график: линии напряженности
    create_field_lines(ax2, lines)
    
    # Общий заголовок
    charge_summary = ', '.join([f'q{i+1}={q:.1f}' for i, (_, _, q) in enumerate(CHARGES)])
    plt.suptitle(f'Моделирование электрического поля ({len(CHARGES)} зарядов: {charge_summary})', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Сохраняем в папку results
    plt.savefig(result_path, dpi=150, bbox_inches='tight')
    print(f"\n5. График сохранен: {result_path}")
    
    print_summary(lines, result_path)
    plt.show()

if __name__ == "__main__":
    main()