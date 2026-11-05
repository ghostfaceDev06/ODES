

function shooting_linear_exemplo
    % Parâmetros do problema
    x0 = 1; 
    xf = 2;
    y0 = 1; 
    yf = 2;
    N = 20;
    h = (xf - x0) / N;

    % Inicialização dos vetores
    x = linspace(x0, xf, N+1);
    u = zeros(2, N+1); % Linha 1: u, Linha 2: u'
    v = zeros(2, N+1); % Linha 1: v, Linha 2: v'

    % Condições iniciais para os dois PVIs
    u(:, 1) = [y0; 0]; % u(1) = 1, u'(1) = 0
    v(:, 1) = [0; 1];  % v(1) = 0, v'(1) = 1

    % Laço do Runge-Kutta de 4ª Ordem (RK4)
    for i = 1:N
        % Passo para o PVI u(x)
        k1_u = h * f_u(x(i), u(:, i));
        k2_u = h * f_u(x(i) + h/2, u(:, i) + k1_u/2);
        k3_u = h * f_u(x(i) + h/2, u(:, i) + k2_u/2);
        k4_u = h * f_u(x(i) + h, u(:, i) + k3_u);
        u(:, i+1) = u(:, i) + (k1_u + 2*k2_u + 2*k3_u + k4_u) / 6;

        % Passo para o PVI v(x)
        k1_v = h * f_v(x(i), v(:, i));
        k2_v = h * f_v(x(i) + h/2, v(:, i) + k1_v/2);
        k3_v = h * f_v(x(i) + h/2, v(:, i) + k2_v/2);
        k4_v = h * f_v(x(i) + h, v(:, i) + k3_v);
        v(:, i+1) = v(:, i) + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6;
    end
close all
    % Cálculo da constante de ajuste c usando a condição de contorno final y(2) = 2
    u_2 = u(1, end);
    v_2 = v(1, end);
    c = (yf - u_2) / v_2;

    % Combinação linear para obter a solução final aproximada
    y_num = u(1, :) + c * v(1, :);

    % Cálculo da Solução Analítica (Exata) para comparação
    c1 = (4/7) * (1.725 + 0.3*sin(log(2)) + 0.1*cos(log(2)));
    c2 = 1.1 - c1;
    y_exact = c1 * x + c2 * (x.^-2) - 0.3 * sin(log(x)) - 0.1 * cos(log(x));


    erro = y_num-y_exact; 

    % Exibição dos resultados em tabela no Command Window
    fprintf('   x_i       u(x_i)      v(x_i)     y_num(x_i)   y_exact(x_i)    erro(i)\n');
    fprintf('-------------------------------------------------------------------------\n');
    for i = 1:N+1
        fprintf('%6.2f   %10.5f  %10.5f  %12.5f  %12.5f   %12.5e\n', ...
                x(i), u(1,i), v(1,i), y_num(i), y_exact(i), abs(y_num(i)-y_exact(i)));
    end

    % Construção do Gráfico
    figure;
    plot(x, y_exact, 'g-', 'LineWidth', 2); hold on;
    plot(x, y_num, 'ro--', 'MarkerFaceColor', 'r');
    grid on;
    title('Solução Numérica via Shooting Linear (N = 10)');
    xlabel('x');
    ylabel('y(x)');
    legend('Solução Analítica (Exata)', 'Shooting Linear (RK4)', 'Location', 'Best');
end

% Função do PVI u (Inclui o termo não homogêneo r(x))
% Y(1) equivale a u, e Y(2) equivale a u'
function dY = f_u(x, Y)
    dY = zeros(2, 1);
    dY(1) = Y(2);
    dY(2) = -(2/x)*Y(2) + (2/x^2)*Y(1) + sin(log(x))/x^2;
end

% Função do PVI v (Equação homogênea correspondente)
% Y(1) equivale a v, e Y(2) equivale a v'
function dY = f_v(x, Y)
    dY = zeros(2, 1);
    dY(1) = Y(2);
    dY(2) = -(2/x)*Y(2) + (2/x^2)*Y(1);
end
