% =========================================================================
% SCRIPT PRINCIPAL: ALGORITMO PREDITOR-CORRETOR DE MILNE-SIMPSON
% =========================================================================
clear; clc; close all;

% --- 1. Definição do Problema de Valor Inicial (PVI) ---
f = @(t, y) y - t.^2 + 1;         % EDO: dy/dt = f(t,y)
y_analitica = @(t) (t + 1).^2 - 0.5 * exp(t); % Solução exata para validação

% --- 2. Parâmetros de Simulação ---
tspan(1) = 0;
tspan(2) = 2;
% Intervalo de tempo [t_inicial, t_final]
y0 = 0.5;                         % Condição inicial y(t0)
h = 0.001;                          % Tamanho do passo de integração

% --- 3. Execução do Método Numérico ---
[t, y_aprox] = predictor_corrector_milne_simpson(f, tspan, y0, h);

% --- 4. Cálculo da Solução Analítica e Erro ---
y_exata = y_analitica(t);
erro_absoluto = abs(y_exata - y_aprox);

% --- 5. Apresentação dos Resultados em Tabela ---
Resultados = table(t', y_aprox', y_exata', erro_absoluto', ...
    'VariableNames', {'Tempo', 'Valor_Aproximado', 'Valor_Exato', 'Erro_Absoluto'});

disp('===================================================================');
disp('         TABELA COMPARATIVA: MILNE-SIMPSON vs SOLUÇÃO EXATA        ');
disp('===================================================================');
disp(Resultados);

% --- 6. Apresentação Gráfica dos Resultados ---
figure('Color', 'w', 'Position', [100, 100, 800, 500]);
plot(t, y_exata, 'g-', 'LineWidth', 2); hold on;
plot(t, y_aprox, 'b^--', 'MarkerFaceColor', 'b', 'MarkerSize', 5, 'LineWidth', 1.2);
grid on;
set(gca, 'FontSize', 11);
legend('Solução Analítica (Exata)', 'Milne-Simpson', 'Location', 'NorthWest');
title('Resolução de EDO via Método Preditor-Corretor de Milne-Simpson', 'FontSize', 13);
xlabel('Tempo (t)', 'FontSize', 12);
ylabel('Solução y(t)', 'FontSize', 12);


% =========================================================================
% FUNÇÃO LOCAL: IMPLEMENTAÇÃO DO MÉTODO DE MILNE-SIMPSON
% =========================================================================
function [t, y] = predictor_corrector_milne_simpson(f, tspan, y0, h)
% Inicialização do vetor de tempo e do vetor de estados
t = tspan(1):h:tspan(2);
N = length(t);
y = zeros(1, N);
y(1) = y0;

% Vetor de histórico para evitar múltiplas avaliações da função f(t,y)
f_hist = zeros(1, N);
f_hist(1) = f(t(1), y(1));

% --- Passo 1: Inicialização (Bootstrap) via Runge-Kutta de 4ª Ordem ---
% O método de Milne requer 4 pontos iniciais conhecidos: y(1), y(2), y(3), y(4)
for i = 1:3
    k1 = f(t(i), y(i));
    k2 = f(t(i) + h/2, y(i) + h*k1/2);
    k3 = f(t(i) + h/2, y(i) + h*k2/2);
    k4 = f(t(i) + h, y(i) + h*k3);

    y(i+1) = y(i) + (h/6) * (k1 + 2*k2 + 2*k3 + k4);
    f_hist(i+1) = f(t(i+1), y(i+1));
end

% --- Passo 2: Iteração do Par Preditor-Corretor ---
for i = 4:(N-1)
    % [PREDITOR] Fórmula explícita de Milne (avança a partir de i-3)
    y_p = y(i-3) + (4*h/3) * (2*f_hist(i) - f_hist(i-1) + 2*f_hist(i-2));

    % Avaliação preliminar da derivada no ponto futuro predito
    f_p = f(t(i+1), y_p);

    % [CORRETOR] Fórmula implícita de Simpson (corrige usando f_p e avalia de i-1)
    y(i+1) = y(i-1) + (h/3) * (f_p + 4*f_hist(i) + f_hist(i-1));

    % Salva a derivada corrigida definitiva no histórico para o próximo passo
    f_hist(i+1) = f(t(i+1), y(i+1));
end
end
