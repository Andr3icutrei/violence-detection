import { ChangeDetectorRef, Component, OnInit } from '@angular/core';
import {TranslatePipe} from "@ngx-translate/core";
import { UsersService } from '../../../../services/users/users-service';
import { UsersStatsResponseDto } from '../../../../core/api/models/users-stats-response-dto';
import { Router } from '@angular/router';

@Component({
  selector: 'app-users-stats',
  imports: [TranslatePipe],
  templateUrl: './users-stats.html',
  styleUrl: './users-stats.css',
})
export class UsersStats implements OnInit {
  usersStats!: UsersStatsResponseDto;

  constructor(
    private cdr: ChangeDetectorRef,
    private router: Router,
    private usersService: UsersService,
  ) {}

  ngOnInit(): void {
    this.loadUserInformation();
  }

  private loadUserInformation(): void {
    this.usersService.getUsersStats().subscribe({
      next: (data: UsersStatsResponseDto) => {
        this.usersStats = data;
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.router.navigate(['/dashboard']);
      },
    });
  }
}
