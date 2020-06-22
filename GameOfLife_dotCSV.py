import numpy as np
import pygame
import time

pygame.init()
# Window size
width, height = 1000, 1000
screen = pygame.display.set_mode((height, width))
# BG color
bg = 25, 25, 25
screen.fill(bg)
# Cell number
nxC, nyC = 100, 100
# Pixels by cell
dimCW = int(width / nxC)
dimCH = int(height / nyC)
# Random init state
gameState = np.random.randint(2, size=(nxC, nyC))
n_neigh = np.zeros(5)
pauseExec = False
# Generation loop
while True:
    # Copy of the game configure depending on game-1
    newGameState = np.copy(gameState)
    screen.fill(bg)
    # time.sleep(0.01)
    # Keyboard event
    ev = pygame.event.get()
    for event in ev:
        if event.type == pygame.KEYDOWN:
            pauseExec = not pauseExec
    # Click event for painting cells
    mouseClick = pygame.mouse.get_pressed()
    if sum(mouseClick) > 0:
        posX, posY = pygame.mouse.get_pos()
        celX, celY = int(np.floor(posX / dimCW)), int(np.floor(posY / dimCH))
        if mouseClick[0] == 1: newGameState[celX, celY] = 1
        if mouseClick[2] == 1: newGameState[celX, celY] = 2

    for y in range(0, nxC):
        for x in range(0, nyC):

            if not pauseExec:

                for c in range(1, 5):
                    n_neigh[c] = 0

                    if gameState[(x - 1) % nxC, (y - 1) % nyC] == c: n_neigh[c] += 1
                    if gameState[x % nxC, (y - 1) % nyC] == c: n_neigh[c] += 1
                    if gameState[(x + 1) % nxC, (y - 1) % nyC] == c: n_neigh[c] += 1
                    if gameState[(x - 1) % nxC, y % nyC] == c: n_neigh[c] += 1
                    if gameState[(x + 1) % nxC, y % nyC] == c: n_neigh[c] += 1
                    if gameState[(x - 1) % nxC, (y + 1) % nyC] == c: n_neigh[c] += 1
                    if gameState[x % nxC, (y + 1) % nyC] == c: n_neigh[c] += 1
                    if gameState[(x + 1) % nxC, (y + 1) % nyC] == c: n_neigh[c] += 1

                    if gameState[x, y] == 0 and n_neigh[c] == 3:
                        newGameState[x, y] = c
                    elif gameState[x, y] == c and (n_neigh[c] < 2 or n_neigh[c] > 3):
                        newGameState[x, y] = 0

                poly = [(x * dimCW, y * dimCH),
                        ((x + 1) * dimCW, y * dimCH),
                        ((x + 1) * dimCW, (y + 1) * dimCH),
                        (x * dimCW, (y + 1) * dimCH)]

                if newGameState[x, y] == 0:
                    pygame.draw.polygon(screen, (128, 128, 128), poly, 1)
                elif newGameState[x, y] == 1:
                    pygame.draw.polygon(screen, (128, 255, 128), poly, 0)
                elif newGameState[x, y] == 2:
                    pygame.draw.polygon(screen, (128, 128, 255), poly, 0)
                else:
                    pygame.draw.polygon(screen, (255, 128, 128), poly, 0)

    gameState = np.copy(newGameState)

    pygame.display.flip()
