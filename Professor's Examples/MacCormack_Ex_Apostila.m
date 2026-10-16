% =========================================================================
% ANALISE DE CONVERGENCIA E MODELAGEM DE CIRCUITO RC VIA PREDITOR-CORRETOR
% =========================================================================
clear; clc;close all


% --- Parametros Fisicos do Circuito ---
V_in = 10.0;        % Tensao de entrada (V)
R = 2000;           % Resistencia (Ohms)
C = 500e-6;         % Capacitancia (Farads)
tau = R * C;        % Constante de tempo (s)

% Defincao da EDO: du/dt = f(t, u)
f = @(t, u) (V_in - u) / tau;

% --- Parametros de Simulacao ---
t_inicial = 0;
t_final = 5 * tau;  % Simular ate a estabilizacao (5 constantes de tempo)
dt = 0.2;           % Passo de tempo discretizado
t = t_inicial:dt:t_final;
Nt = length(t);

% --- Alocacao de Memoria ---
u = zeros(1, Nt);
u(1) = 0.0;         % Condicao inicial: capacitor descarregado

% --- Loop de Integracao Numerica ---
for n = 1:(Nt-1)
    % 1. ESTADO PREDITO (Avanço com derivada em t_n)
    f_local = f(t(n), u(n));
    u_pred = u(n) + dt * f_local;
    
    % 2. ESTADO CORRIGIDO (Derivada no ponto t_n+1 predito)
    f_futuro = f(t(n+1), u_pred);
    u_corr = u(n) + dt * f_futuro;
    
    % 3. ATUALIZACAO COMPUTACOES VIA MEDIA ARITMETICA
    u(n+1) = 0.5 * (u_pred + u_corr);
end

% --- Solucao Analitica (Exata) ---
u_exata = V_in * (1 - exp(-t / tau));

% --- Calculo do Erro Absoluto ---
erro_absoluto = abs(u_exata - u);

% --- Graficos de Desempenho Academico ---
figure('Color', 'w');

% Subplot 1: Resposta Temporal
subplot(1, 2, 1);
plot(t, u_exata, '-k', 'LineWidth', 2); hold on;
plot(t, u, '--or', 'MarkerFaceColor', 'r', 'MarkerSize', 4);
grid on;
xlabel('Tempo (segundos)');
ylabel('Tensão no Capacitor V_c (V)');
title('Resposta Transiente do Circuito RC');
legend('Solução Exata', 'Preditor-Corretor (MacCormack)', 'Location', 'SouthEast');

% Subplot 2: Perfil do Erro Absoluto ao Longo do Tempo
subplot(1, 2, 2);
plot(t, erro_absoluto, '-b', 'LineWidth', 1.5);
grid on;
xlabel('Tempo (segundos)');
ylabel('Erro Absoluto |V_{exata} - V_{num}| (V)');
title('Evolução do Erro de Truncamento Global');
