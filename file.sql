create table admins
(
    id       int auto_increment
        primary key,
    username varchar(50)  not null,
    password varchar(100) not null,
    constraint username
        unique (username)
);

create table books
(
    id          int auto_increment
        primary key,
    name        varchar(255)                        not null,
    author      varchar(255)                        not null,
    summary     text                                not null,
    description text                                not null,
    pdf_path    varchar(255)                        not null,
    image_path  varchar(255)                        not null,
    created_at  timestamp default CURRENT_TIMESTAMP null
);

create table users
(
    id         int auto_increment
        primary key,
    name       varchar(100)                       not null,
    email      varchar(100)                       not null,
    username   varchar(50)                        not null,
    password   varchar(255)                       not null,
    created_at datetime default CURRENT_TIMESTAMP null,
    constraint email
        unique (email),
    constraint username
        unique (username)
);

