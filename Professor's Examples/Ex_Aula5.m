% =========================================================================
% MÉTODOS DE ADAMS-BASHFORTH, ADAMS-MOULTON E TABELA COMPARATIVA INTEGRADA
% =========================================================================
clear; clc; close all;

% 1. Definição do Problema (dy/dt = f(t,y))
f = @(t, y) y - t^2 + 1; 
y_exata = @(t) (t + 1).^2 - 0.5*exp(t);

t0 = 0;       % Tempo inicial
tf = 2;       % Tempo final
y0 = 0.5;       % Condição inicial y(t0)
h = 0.1;      % Tamanho do passo

% Vetor de tempo e número de pontos
t = t0:h:tf;
N = length(t);

% 2. Inicialização dos Vetores de Solução
y_ab2 = zeros(1, N); y_ab3 = zeros(1, N); y_ab4 = zeros(1, N); y_ab5 = zeros(1, N);
y_am2 = zeros(1, N); y_am3 = zeros(1, N); y_am4 = zeros(1, N);

% 3. Inicialização dos Primeiros Passos usando RK4 (até 5 pontos)
y_init = zeros(1, 5);
y_init(1) = y0;

for i = 1:4
    k1 = f(t(i), y_init(i));
    k2 = f(t(i) + h/2, y_init(i) + h*k1/2);
    k3 = f(t(i) + h/2, y_init(i) + h*k2/2);
    k4 = f(t(i) + h, y_init(i) + h*k3);
    y_init(i+1) = y_init(i) + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
end

% Atribuição dos pontos iniciais para todos os métodos
y_ab2(1:2) = y_init(1:2); y_am2(1:2) = y_init(1:2);
y_ab3(1:3) = y_init(1:3); y_am3(1:3) = y_init(1:3);
y_ab4(1:4) = y_init(1:4); y_am4(1:4) = y_init(1:4);
y_ab5(1:5) = y_init(1:5);

% =========================================================================
% 4. LOOPS DE INTEGRAÇÃO
% =========================================================================

% --- ADAMS-BASHFORTH (AB) ---
for n = 2:N-1
    y_ab2(n+1) = y_ab2(n) + (h/2) * (3*f(t(n), y_ab2(n)) - f(t(n-1), y_ab2(n-1)));
end
for n = 3:N-1
    y_ab3(n+1) = y_ab3(n) + (h/12) * (23*f(t(n), y_ab3(n)) - 16*f(t(n-1), y_ab3(n-1)) + 5*f(t(n-2), y_ab3(n-2)));
end
for n = 4:N-1
    y_ab4(n+1) = y_ab4(n) + (h/24) * (55*f(t(n), y_ab4(n)) - 59*f(t(n-1), y_ab4(n-1)) + 37*f(t(n-2), y_ab4(n-2)) - 9*f(t(n-3), y_ab4(n-3)));
end
for n = 5:N-1
    y_ab5(n+1) = y_ab5(n) + (h/720) * (1901*f(t(n), y_ab5(n)) - 2774*f(t(n-1), y_ab5(n-1)) + 2616*f(t(n-2), y_ab5(n-2)) - 1274*f(t(n-3), y_ab5(n-3)) + 251*f(t(n-4), y_ab5(n-4)));
end

% --- ADAMS-MOULTON (AM) ---
for n = 2:N-1
    f_n = f(t(n), y_am2(n)); f_nm1 = f(t(n-1), y_am2(n-1));
    y_predict = y_am2(n) + (h/2) * (3*f_n - f_nm1);
    y_am2(n+1) = y_am2(n) + (h/12) * (5*f(t(n+1), y_predict) + 8*f_n - f_nm1);
end
for n = 3:N-1
    f_n = f(t(n), y_am3(n)); f_nm1 = f(t(n-1), y_am3(n-1)); f_nm2 = f(t(n-2), y_am3(n-2));
    y_predict = y_am3(n) + (h/12) * (23*f_n - 16*f_nm1 + 5*f_nm2);
    y_am3(n+1) = y_am3(n) + (h/24) * (9*f(t(n+1), y_predict) + 19*f_n - 5*f_nm1 + f_nm2);
end
for n = 4:N-1
    f_n = f(t(n), y_am4(n)); f_nm1 = f(t(n-1), y_am4(n-1)); f_nm2 = f(t(n-2), y_am4(n-2)); f_nm3 = f(t(n-3), y_am4(n-3));
    y_predict = y_am4(n) + (h/24) * (55*f_n - 59*f_nm1 + 37*f_nm2 - 9*f_nm3);
    y_am4(n+1) = y_am4(n) + (h/720) * (251*f(t(n+1), y_predict) + 646*f_n - 264*f_nm1 + 106*f_nm2 - 19*f_nm3);
end

% =========================================================================
% 5. CRIAÇÃO E EXIBIÇÃO DA TABELA NO TERMINAL
% =========================================================================
y_real_tf = y_exata(tf);

% Arrays de dados para compor a tabela
Metodo = {'Adams-Bashforth 2'; 'Adams-Bashforth 3'; 'Adams-Bashforth 4'; ...
          'Adams-Bashforth 5'; 'Adams-Moulton 2'; 'Adams-Moulton 3'; ...
          'Adams-Moulton 4'};

Valor_Aproximado = [y_ab2(end); y_ab3(end); y_ab4(end); y_ab5(end); ...
                    y_am2(end); y_am3(end); y_am4(end)];

Erro_Absoluto = abs(y_real_tf - Valor_Aproximado);

% Comando nativo que gera o objeto estruturado 'table'
Tabela_Resultados = table(Metodo, Valor_Aproximado, Erro_Absoluto);

% Print formatado no Command Window
fprintf('\n=================================================================\n');
fprintf('       TABELA COMPARATIVA NO TEMPO FINAL (t = %.2f, Valor Real = %.6f)\n', tf, y_real_tf);
fprintf('=================================================================\n');
disp(Tabela_Resultados);

% =========================================================================
% 6. VISUALIZAÇÃO GRÁFICA
% =========================================================================
figure('Position', [100, 100, 1100, 500]);

subplot(1, 2, 1);
plot(t, y_exata(t), 'k-', 'LineWidth', 2); hold on;
plot(t, y_ab2, 'r--o', 'MarkerFaceColor', 'r');
plot(t, y_ab3, 'b--s', 'MarkerFaceColor', 'b');
plot(t, y_ab4, 'm--^', 'MarkerFaceColor', 'm');
plot(t, y_ab5, 'g--*', 'MarkerFaceColor', 'g');
grid on; xlabel('Tempo (t)'); ylabel('y(t)');
title('Adams-Bashforth (Explícitos)');
legend('Solução Exata', 'AB 2', 'AB 3', 'AB 4', 'AB 5', 'Location', 'Best');

subplot(1, 2, 2);
plot(t, y_exata(t), 'k-', 'LineWidth', 2); hold on;
plot(t, y_am2, 'g--o', 'MarkerFaceColor', 'c');
plot(t, y_am3, 'y--s', 'MarkerFaceColor', 'y');
plot(t, y_am4, 'b--^', 'MarkerFaceColor', 'm');
grid on; xlabel('Tempo (t)'); ylabel('y(t)');
title('Adams-Moulton (Implícito)');
legend('Solução Exata', 'AM 2', 'AM 3', 'AM 4', 'Location', 'Best');
