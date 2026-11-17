% =========================================================================
% SCRIPT MATLAB: RKF45 COM COMPARAÇÃO ANALÍTICA (CÓDIGO CONTÍNUO)
% =========================================================================

clear; clc; close all;

% --- 1. DEFINIÇÃO DO PROBLEMA (EDO E SOLUÇÃO ANALÍTICA) ---
f = @(t, y) y - t^2 + 1;
y_analitica = @(t) (t + 1).^2 - 0.5 * exp(t); % Solução exata/analítica

% --- 2. CONDIÇÕES INICIAIS E INTERVALO DE TEMPO ---
t0 = 0;         % Tempo inicial
tf = 2;         % Tempo final
y0 = 0.5;       % Condição inicial y(t0)

% --- 3. PARÂMETROS DE TOLERÂNCIA E LIMITES DE PASSO ---
tol = 1e-5;     % Tolerância de erro aceitável
hmax = 0.25;    % Passo MÁXIMO permitido
hmin = 0.01;    % Passo MÍNIMO permitido


% --- 4. INICIALIZAÇÃO DE VARIÁVEIS ---
t = t0;
y = y0;
h = 0.25;       % Inicia com o passo máximo permitido
i = 1;          % Contador de passos do loop

% --- 5. LOOP PRINCIPAL DE INTEGRAÇÃO (RKF45) ---
while t(i) < tf
    % Ajusta o passo se ele for ultrapassar o tempo final estipulado
    if t(i) + h > tf
        h = tf - t(i);
    end
    
    % Coeficientes de Runge-Kutta-Fehlberg (RKF45)
    k1 = h * f(t(i), y(i));
    k2 = h * f(t(i) + h/4, y(i) + k1/4);
    k3 = h * f(t(i) + 3*h/8, y(i) + 3*k1/32 + 9*k2/32);
    k4 = h * f(t(i) + 12*h/13, y(i) + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197);
    k5 = h * f(t(i) + h, y(i) + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104);
    k6 = h * f(t(i) + h/2, y(i) - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40);
    
    % Estimativa da solução por aproximação de 4ª Ordem
    y_next = y(i) + 25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5;
    
    % Estimativa do erro local por unidade de passo (R)
    R = abs(1/360*k1 - 128/4275*k3 - 2197/75240*k4 + 1/50*k5 + 2/55*k6) / h;
    
    % Fator de escala multiplicador para ajuste do passo adaptativo
    if R == 0
        s = 2;
    else
        s = 0.84 * (tol / R)^(1/4);
    end
    
    % Se o erro for tolerável OU se o passo já estiver esmagado no mínimo (0.01)
    if (R <= tol) || (h <= hmin)
        if (R > tol) && (h <= hmin)
            warning('Aviso: Erro estimando (%e) maior que a tolerância em t = %f, mas o passo travou em h_min (0.01).', R, t(i));
        end
        % Salva o avanço no vetor de soluções
        t(i+1) = t(i) + h;
        y(i+1) = y_next;
        i = i + 1;
    end
    
    % Atualiza o tamanho do passo adaptativo para a próxima iteração
    h = s * h;
    
    % Garante restrição absoluta dentro da janela regulamentada [0.01 , 0.25]
    if h > hmax
        h = hmax;
    elseif h < hmin
        h = hmin;
    end
end

% --- 6. CÁLCULO DA SOLUÇÃO ANALÍTICA E ERRO REAL ---
y_exato = y_analitica(t);
erro_absoluto = abs(y - y_exato);

% --- 7. EXIBIÇÃO DOS RESULTADOS NO TERMINAL ---
disp('   Tempo (t)   |   RKF45 (y)   |  Analítica (y) |   Erro Absoluto');
disp('-------------------------------------------------------------------');
for j = 1:length(t)
    fprintf('  %10.5f   |  %12.6f |  %12.6f  |   %11.3e\n', t(j), y(j), y_exato(j), erro_absoluto(j));
end

% --- 8. GRÁFICO DE COMPARAÇÃO ---
figure;
subplot(2,1,1); % Gráfico superior: Soluções
plot(t, y, '-', 'MarkerFaceColor', 'r', 'DisplayName', 'RKF45 (Numérico)');
hold on;
% Linha contínua fina para a analítica usando mais pontos para ficar suave
t_suave = linspace(t0, tf, 200);
plot(t_suave, y_analitica(t_suave), '--', 'LineWidth', 1.5, 'DisplayName', 'Analítica (Exata)');
grid on;
xlabel('Tempo (t)'); ylabel('y(t)');
title('Comparação: Solução Numérica RKF45 vs Solução Analítica');
legend('Location', 'northwest');

subplot(2,1,2); % Gráfico inferior: Erro Real
stem(t, erro_absoluto, 'g', 'LineWidth', 1.2, 'MarkerFaceColor', 'k');
grid on;
xlabel('Tempo (t)'); ylabel('Erro Absoluto');
title('Erro Absoluto Real ao Longo do Tempo');
