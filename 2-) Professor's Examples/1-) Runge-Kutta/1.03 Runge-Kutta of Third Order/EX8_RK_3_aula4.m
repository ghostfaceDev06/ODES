
clc
clear
close all
% Parâmetros iniciais
t0 = 1;
tf = 1.5;
h = 0.1;
t = t0:h:tf;
n = length(t);

% Função EDO: y' = f(t, y)
f = @(t, y) 2*t*y;

% Solução exata para comparação
y_exata = @(t) exp(t.^2-1);

% Inicialização dos vetores
y_RK3 = zeros(1, n);


y_RK3(1) = 1;

% Implementação do Método de Runge Kutta de Quarta Ordem 
for i = 1:n-1
    % Método de Runge Kutta
    f_1 = h*f(t(i), y_RK3(i));
    f_2 = h*f(t(i)+(1/2)*h, y_RK3(i)+(h/2)*f_1);
    f_3 = h*f(t(i)+ h, y_RK3(i)+2*h*f_2-h*f_1);
    y_RK3(i+1) = y_RK3(i) + (1/6) * (f_1 + 4*f_2 + f_3);
    
end

% Gráfico comparativo
figure;
plot(t, y_exata(t), 'k-', 'LineWidth', 2); hold on;
plot(t, y_RK3, 'ro--', 'LineWidth', 1.5);
xlabel('Tempo (t)');
ylabel('Valor de y');
legend('Exata', 'RK_3', 'Location', 'Best');
title('Comparação: Solução Exata vs Runge-Kutta de Terceira Ordem');
grid on;

u = y_exata(t)
erro_absoluto=abs(y_RK3 - u)


for i = 1:n
   erro_percentual(i)=abs(erro_absoluto(i)/y_RK3(i))*100;
end
erro_percentual;

format long

A = [y_RK3' u' erro_percentual'];

T = array2table(A,VariableNames=["RK_3" "Exata" "Erro Percentual"])
