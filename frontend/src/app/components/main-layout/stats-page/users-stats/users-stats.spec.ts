import { ComponentFixture, TestBed } from '@angular/core/testing';

import { UsersStats } from './users-stats';

describe('UsersStats', () => {
  let component: UsersStats;
  let fixture: ComponentFixture<UsersStats>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [UsersStats],
    }).compileComponents();

    fixture = TestBed.createComponent(UsersStats);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
