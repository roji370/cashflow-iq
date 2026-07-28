FROM node:22-alpine

WORKDIR /app

# Install dependencies
COPY package.json ./
RUN npm install

# Source code is bind-mounted at runtime via docker-compose
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
